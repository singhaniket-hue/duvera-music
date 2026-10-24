"""Controlled separator fixtures test the gate; no model download or vocal classification is claimed."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from scripts.create_demo import create_demo
from src.director import rules_plan
from src.dsp import transient_exit
from src.media import PipelineError, SR, write_audio
from src.models import RunConfig, Segment, Transcript
from src.music_policy import MANDATORY_INSTRUMENTAL_SUFFIX, instrumental_prompt
from src.narrative import fully_silent, protect_narrative
from src.pipeline import execute
from src.providers import AssetStore, generation_jobs
from src.providers.elevenlabs import ElevenLabsProvider
from src.vocals import VocalGuard, read_float, vocal_scores


class ControlledSeparator:
    """Returns known energy fractions; this fixture is not an actual speech/singing model."""
    def __init__(self, fraction=0.001, residual=0.001):
        self.fraction, self.residual = fraction, residual
        self.calls = self.preparations = 0

    def prepare(self):
        self.preparations += 1

    def separate(self, path, output):
        self.calls += 1
        data, rate = read_float(path)
        fraction = self.residual if "residual" in path.parts else self.fraction
        output.mkdir(parents=True, exist_ok=True)
        vocal, backing = output / "vocals.wav", output / "no_vocals.wav"
        wavfile.write(vocal, rate, data * fraction)
        wavfile.write(backing, rate, data * (1 - fraction))
        return vocal, backing


class InstrumentalContracts(unittest.TestCase):
    def test_conflicting_prompt_is_sanitized_idempotently_and_bounded(self):
        prompt = "melancholic song, male singer with lyrics, sparse piano; female vocal chops"
        safe = instrumental_prompt(prompt)
        self.assertNotIn("male singer", safe)
        self.assertNotIn("female vocal chops", safe)
        self.assertIn("sparse piano", safe)
        self.assertTrue(safe.endswith(MANDATORY_INSTRUMENTAL_SUFFIX))
        self.assertEqual(safe, instrumental_prompt(safe))
        self.assertLessEqual(len(instrumental_prompt("piano " * 2000)), 4000)

    def test_direct_provider_payload_keeps_flag_and_hard_suffix(self):
        _, body = ElevenLabsProvider.payload("male R&B singer, emotional song", 15, "music")
        self.assertTrue(body["force_instrumental"])
        self.assertTrue(body["prompt"].endswith(MANDATORY_INSTRUMENTAL_SUFFIX))
        self.assertNotIn("male R&B singer", body["prompt"])

    def test_asset_store_enforces_contract_on_a_provider_that_ignores_planner(self):
        class Provider:
            prompt = None
            def generate(self, prompt, duration, kind, target):
                self.prompt = prompt
                write_audio(target, np.ones((3 * SR, 2), np.float32) * 0.01)
        with tempfile.TemporaryDirectory() as temp:
            provider = Provider()
            config = RunConfig(script_path=Path("fixture.txt"))
            store = AssetStore(Path(temp), config, None, provider)
            store.get("cue_001", "soulful singing, piano", 3, "music")
            self.assertTrue(provider.prompt.endswith(MANDATORY_INSTRUMENTAL_SUFFIX))


class VocalProxyGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="duvera vocal gate ")
        self.root = Path(self.temp.name)
        time = np.arange(2 * SR) / SR
        self.music = np.column_stack([0.1 * np.sin(2 * np.pi * 440 * time)] * 2).astype(np.float32)
        self.path = self.root / "source.wav"
        write_audio(self.path, self.music)

    def tearDown(self):
        self.temp.cleanup()

    def guard(self, policy="reject", separator=None):
        config = RunConfig(script_path=Path("fixture.txt"), vocal_policy=policy)
        return VocalGuard(config, self.root / policy, separator or ControlledSeparator())

    def test_low_proxy_pass_and_content_deduplication(self):
        separator = ControlledSeparator()
        guard = self.guard(separator=separator)
        guard.prepare()
        self.assertEqual(guard.inspect("cue_001", self.path), self.path)
        self.assertEqual(guard.inspect("cue_002", self.path), self.path)
        self.assertEqual(separator.calls, 1)
        self.assertTrue(guard.reports["cue_002"]["reused_content_check"])

    def test_suspect_asset_rejected_and_original_unchanged(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        guard = self.guard(separator=ControlledSeparator(0.25))
        with self.assertRaisesRegex(PipelineError, "vocal-stem proxy"):
            guard.inspect("cue_002", self.path)
        self.assertEqual(guard.reports["cue_002"]["status"], "rejected")
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertTrue((self.root / "reject/vocal_qc.json").exists())

    def test_strip_rechecks_music_without_touching_original(self):
        separator = ControlledSeparator(0.25, residual=0.001)
        guard = self.guard("strip", separator)
        selected = guard.inspect("cue_002", self.path)
        self.assertNotEqual(selected, self.path)
        self.assertTrue(selected.is_file())
        self.assertEqual(separator.calls, 2)
        self.assertEqual(guard.reports["cue_002"]["status"], "stripped_and_rechecked")
        audio, rate = read_float(selected)
        self.assertEqual((len(audio), rate), (len(self.music), SR))

    def test_residual_failure_is_not_silently_accepted(self):
        guard = self.guard("strip", ControlledSeparator(0.25, residual=0.25))
        with self.assertRaisesRegex(PipelineError, "still exceeds"):
            guard.inspect("cue_002", self.path)
        self.assertEqual(guard.reports["cue_002"]["status"], "rejected_residual")

    def test_proxy_is_level_invariant_and_catches_short_burst(self):
        vocal = np.zeros_like(self.music)
        vocal[SR:SR + SR // 4] = self.music[SR:SR + SR // 4] * 0.4
        a = vocal_scores(self.music, vocal, SR)
        b = vocal_scores(self.music * 0.01, vocal * 0.01, SR)
        key = "max_window_vocal_at_minus18_mix_dbfs"
        self.assertAlmostEqual(a[key], b[key], places=4)
        self.assertGreater(a[key], -35)

    def test_checker_failure_is_fail_closed_and_reported(self):
        separator = ControlledSeparator()
        with patch.object(separator, "separate", side_effect=PipelineError("model failed")):
            guard = self.guard(separator=separator)
            with self.assertRaisesRegex(PipelineError, "model failed"):
                guard.inspect("cue_002", self.path)
        self.assertEqual(guard.reports["cue_002"]["status"], "checker_error")

    def test_off_is_explicitly_unchecked(self):
        guard = self.guard("off")
        self.assertEqual(guard.inspect("cue_001", self.path), self.path)
        self.assertEqual(guard.reports["cue_001"]["status"], "disabled_unchecked")

    def test_cached_generation_is_rechecked_without_paid_regeneration(self):
        music = self.music
        class Provider:
            calls = 0
            def generate(self, prompt, duration, kind, destination):
                self.calls += 1
                write_audio(destination, music)
        provider = Provider()
        config = RunConfig(script_path=Path("fixture.txt"))
        store = AssetStore(self.root / "generated", config, None, provider)
        guard = self.guard(separator=ControlledSeparator(0.25))
        for _ in range(2):
            raw = store.get("cue_001", "sparse piano", 3, "music")
            with self.assertRaisesRegex(PipelineError, "vocal-stem proxy"):
                guard.inspect("cue_001", raw)
        self.assertEqual(provider.calls, 1)

    def test_missing_checker_blocks_before_any_paid_provider_initializes(self):
        script = self.root / "script.txt"
        script.write_text("A short narration with a clear musical bed.")
        config = RunConfig(script_path=script, mode="assets_and_script", provider="elevenlabs",
                           allow_paid_generation=True, output_dir=self.root / "preflight")
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-only-key"}):
            with patch("src.vocals.importlib.util.find_spec", return_value=None):
                with patch("src.pipeline.ElevenLabsProvider") as provider:
                    with self.assertRaisesRegex(PipelineError, "vocals extra"):
                        execute(config)
                    provider.assert_not_called()

    def test_imported_suspect_music_cannot_reach_renderer(self):
        create_demo(self.root / "demo", seconds=6)
        config = RunConfig(video_path=self.root / "demo/raw.mp4",
                           transcript_path=self.root / "demo/transcript.json",
                           local_assets=self.root / "demo/assets.json", provider="local",
                           mode="final_video", cue_seconds=3, output_dir=self.root / "blocked")
        with patch("src.vocals.DemucsSeparator", return_value=ControlledSeparator(0.25)):
            with patch("src.pipeline.render") as renderer:
                with self.assertRaisesRegex(PipelineError, "vocal-stem proxy"):
                    execute(config)
                renderer.assert_not_called()
        report = json.loads((config.output_dir / "vocal_checks/vocal_qc.json").read_text())
        self.assertTrue(any(x["status"] == "rejected" for x in report["assets"].values()))


class EditorialPolish(unittest.TestCase):
    def story(self):
        return Transcript(segments=[Segment(start=0, end=68, text="A bright musical morning."),
                                    Segment(start=68, end=82, text="Finally realize kiya something funny."),
                                    Segment(start=82, end=86, text="Agar aap bhi meri tarah hain hajam nahi hoga.")])

    def test_outro_reuses_first_music_without_new_generation(self):
        from src.models import VisualEvent
        config = RunConfig(script_path=Path("fixture.txt"), cue_seconds=120, outro_return_sec=82)
        visual = VisualEvent(start_sec=68, end_sec=86, kind="black_card_candidate",
                             mean_luma=3, black_pixel_ratio=0.98, bright_pixel_ratio=0.01)
        sheet = rules_plan(self.story(), 86, config, visuals=[visual])
        outro = next(c for c in sheet.cues if c.start_sec == 82)
        self.assertEqual(outro.reuse_music_from, sheet.cues[0].cue_id)
        self.assertNotIn(outro.cue_id, [c.cue_id for c, kind, _, _ in generation_jobs(sheet) if kind == "music"])

    def test_cta_shortens_preexisting_llm_silence_and_can_be_disabled(self):
        from src.models import SilenceWindow, VisualEvent
        config = RunConfig(script_path=Path("fixture.txt"), cue_seconds=120)
        base = rules_plan(self.story(), 86, config)
        base.silence_windows = [SilenceWindow(start_sec=68, end_sec=86, reason="LLM broad silence")]
        visual = VisualEvent(start_sec=68, end_sec=86, kind="black_card_candidate",
                             mean_luma=3, black_pixel_ratio=0.98, bright_pixel_ratio=0.01)
        result = protect_narrative(base, self.story(), [visual], config)
        self.assertFalse(fully_silent(82, 86, result.silence_windows))
        config.outro_music_return = False
        result = protect_narrative(base, self.story(), [visual], config)
        self.assertTrue(fully_silent(68, 86, result.silence_windows))

    def test_exit_prefers_real_pluck_half_second_before_black_boundary(self):
        audio = np.zeros((3 * SR, 2), np.float32)
        for time, level in ((2.22, 0.3), (2.5, 0.2)):
            start = round(time * SR)
            audio[start:start + round(0.09 * SR)] = level * np.exp(-np.arange(round(0.09 * SR)) / 2000)[:, None]
        output, record = transient_exit(audio, 200, 0.5)
        self.assertAlmostEqual(record["onset_sec"], 2.5, delta=0.03)
        self.assertEqual(abs(output[-500:]).max(), 0)


if __name__ == "__main__":
    unittest.main()
