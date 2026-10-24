"""Regression tests exercise real FFmpeg renders; provider networking is mocked or absent."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from scripts.create_demo import create_demo
from src.analyzer import estimated_transcript
from src.director import rules_plan
from src.dsp import duck, fit_music, silence_envelope
from src.linguistics import syllables, whisper_options
from src.media import PipelineError, SR, probe, read_audio
from src.models import RunConfig, Segment, Transcript
from src.pipeline import _gate, execute
from src.preflight import check_environment
from src.providers import AssetStore
from src.providers.elevenlabs import ElevenLabsProvider


class ContractsAndDSP(unittest.TestCase):
    def config(self):
        return RunConfig(script_path=Path("script.txt"))

    def sheet(self):
        transcript = Transcript(segments=[Segment(start=0, end=3, text="One short line."),
                                          Segment(start=3, end=6, text="The truth matters.")])
        return rules_plan(transcript, 6, self.config())

    def test_hindi_and_roman_counts_are_explicit_estimates(self):
        self.assertEqual(syllables("आपने", "hinglish"), 2)
        self.assertEqual(syllables("aapne", "hinglish"), 2)
        self.assertEqual(syllables("people"), 2)
        self.assertEqual(syllables("किताब", "hindi"), 2)
        self.assertIsNone(whisper_options("hinglish")["language"])

    def test_invalid_and_nonfinite_times_rejected(self):
        for end in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValidationError):
                Segment(start=1, end=end, text="test")
        with self.assertRaises(ValidationError):
            Transcript(segments=[Segment(start=2, end=4, text="x"), Segment(start=1, end=2, text="y")])

    def test_estimated_timing_warning(self):
        transcript = estimated_transcript("A simple sentence. Another useful idea.", 140)
        sheet = rules_plan(transcript, transcript.segments[-1].end, self.config())
        self.assertEqual(sheet.timing_source, "estimated")
        self.assertTrue(any("NOT aligned" in s for s in sheet.warnings))

    def test_short_audio_loop_has_exact_length_and_click_free_exit(self):
        source = np.ones((500, 2), dtype=np.float32) * 0.1
        for duration in (0.001, 0.03, 0.5):
            result = fit_music(source, duration)
            self.assertEqual(len(result), round(duration * SR))
            self.assertLess(abs(result[-1]).max(), 1e-6)
            self.assertTrue(np.isfinite(result).all())

    def test_silence_is_zero_and_dialogue_untouched(self):
        sheet = self.sheet()
        gain = silence_envelope(6 * SR, 0, sheet)
        self.assertEqual(gain[round(3.2 * SR)], 0)
        self.assertEqual(gain[SR], 1)

    def test_measured_ducking_reduces_dense_music(self):
        t = np.arange(3 * SR) / SR
        music = np.column_stack([0.25 * np.sin(2 * np.pi * 700 * t)] * 2)
        voice = np.column_stack([0.1 * np.sin(2 * np.pi * 1000 * t)] * 2)
        output, qc = duck(music, voice, self.sheet().cues[0])
        self.assertLess(np.mean(output ** 2), np.mean(music ** 2) / 10)
        self.assertGreater(qc["measured_gap_p10_lu"], 10)

    def test_provider_endpoints_separate_music_and_sfx(self):
        endpoint, body = ElevenLabsProvider.payload("instrumental", 15, "music")
        self.assertTrue(endpoint.startswith("/music?"))
        self.assertTrue(body["force_instrumental"])
        self.assertEqual(body["music_length_ms"], 15000)
        endpoint, body = ElevenLabsProvider.payload("whoosh", 2, "sfx")
        self.assertTrue(endpoint.startswith("/sound-generation?"))
        self.assertNotIn("force_instrumental", body)

    def test_gate_validates_both_before_spend(self):
        config = RunConfig(script_path=Path("script.txt"), mode="assets_and_script",
                           provider="both_compare", allow_paid_generation=True)
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-secret"}):
            with self.assertRaisesRegex(PipelineError, "Google Flow"):
                _gate(config, self.sheet())
        config.max_generated_seconds = 1
        with self.assertRaisesRegex(PipelineError, "exceeding"):
            _gate(config, self.sheet())

    def test_uncertain_paid_generation_not_retried(self):
        class Failing:
            count = 0
            def generate(self, *args):
                self.count += 1
                raise PipelineError("transport interrupted")
        with tempfile.TemporaryDirectory() as temp:
            provider = Failing()
            store = AssetStore(Path(temp), self.config(), None, provider)
            with self.assertRaisesRegex(PipelineError, "transport"):
                store.get("cue_001", "test", 3, "music")
            with self.assertRaisesRegex(PipelineError, "Uncertain"):
                store.get("cue_001", "test", 3, "music")
            self.assertEqual(provider.count, 1)

    def test_non_tty_never_waits_for_input_and_preflight_no_secret(self):
        result = subprocess.run([sys.executable, "-m", "src.cli"], input="", capture_output=True,
                                text=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "unique-sensitive-secret"}):
            env = json.dumps(check_environment())
            self.assertNotIn("unique-sensitive-secret", env)
            self.assertIn("not_verified", env)


class RealMediaIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="duvera integration ")
        cls.root = Path(cls.temp.name)
        create_demo(cls.root, seconds=6)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def config(self, **changes):
        data = dict(video_path=self.root / "raw.mp4", transcript_path=self.root / "transcript.json",
                    mode="final_video", provider="local", local_assets=self.root / "assets.json",
                    cue_seconds=3, output_dir=self.root / "render", vocal_policy="off")
        data.update(changes)
        return RunConfig(**data)

    def test_actual_video_encoded_qc_silence_and_timeline(self):
        config = self.config()
        summary = execute(config)
        self.assertEqual(summary["status"], "complete")
        branch = config.output_dir / "local"
        report = json.loads((branch / "qc_report.json").read_text())
        self.assertTrue(report["mastering"]["passed"])
        self.assertLessEqual(report["mastering"]["encoded_measurement"]["input_tp"], -1)
        info = probe(branch / report["mastering"]["video"])
        self.assertAlmostEqual(float(info["format"]["duration"]), 6, delta=0.1)
        bed, sr = read_audio(branch / "music.wav")
        self.assertEqual(sr, SR)
        self.assertEqual(len(bed), 6 * SR)
        self.assertEqual(abs(bed[round(2.1 * SR):round(3 * SR)]).max(), 0)
        xml = ET.parse(branch / "timeline.xml")
        self.assertTrue(xml.findall(".//clipitem"))
        self.assertTrue((branch / "markers.json").exists())
        # Resume retains the same plan and succeeds without a provider call.
        self.assertEqual(execute(config)["status"], "complete")
        config.target_lufs = -16
        with self.assertRaisesRegex(PipelineError, "changed"):
            execute(config)
        self.assertFalse((config.output_dir / ".run.lock").exists())

    def test_comparison_imports_both_without_credentials(self):
        config = self.config(provider="both_compare", local_assets=None,
                             elevenlabs_assets=self.root / "assets.json",
                             google_flow_assets=self.root / "assets.json",
                             mode="assets_and_script", output_dir=self.root / "compare")
        with patch.dict("os.environ", {}, clear=True):
            summary = execute(config)
        self.assertEqual(summary["providers"], ["elevenlabs", "google_flow"])
        self.assertTrue((config.output_dir / "ab_comparison_report.md").exists())
        a, _ = read_audio(config.output_dir / "elevenlabs" / "music.wav")
        b, _ = read_audio(config.output_dir / "google_flow" / "music.wav")
        np.testing.assert_array_equal(a, b)

    def test_long_transcript_fails_before_generation(self):
        transcript = self.root / "out_of_bounds.json"
        transcript.write_text(json.dumps({"segments": [{"start": 0, "end": 99, "text": "invalid"}]}))
        with self.assertRaisesRegex(PipelineError, "beyond audio"):
            execute(self.config(transcript_path=transcript, output_dir=self.root / "invalid"))

    def test_output_lock_blocks_concurrent_generation(self):
        root = self.root / "locked"
        root.mkdir(exist_ok=True)
        (root / ".run.lock").write_text("other process")
        with self.assertRaisesRegex(PipelineError, "locked"):
            execute(self.config(output_dir=root))


if __name__ == "__main__":
    unittest.main()
