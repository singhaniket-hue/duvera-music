"""Commentary preset tests use controlled speaker scores and real FFmpeg, never a live voice model."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.create_demo import create_demo
from src.archival import ArchivalInsertWindow, detect_reference_windows, mismatch_windows, protect_archival
from src.dialogue_polish import polish_dialogue
from src.director import plan, rules_plan
from src.documentary import chapter_summary
from src.dsp import silence_envelope
from src.event_scheduler import schedule
from src.media import PipelineError, SR, read_audio, write_audio
from src.models import RunConfig, Segment, Transcript
from src.pacing_models import EditEvent, EventGraph
from src.pipeline import execute
from src.providers import generation_jobs


class CommentaryPolicy(unittest.TestCase):
    def test_preset_defaults_and_explicit_overrides(self):
        config = RunConfig(script_path=Path("script.txt"), preset="social_commentary")
        self.assertTrue(config.dialogue_polish)
        self.assertEqual(config.sfx_mode, "accents_only")
        self.assertFalse(config.outro_music_return)
        explicit = RunConfig(script_path=Path("script.txt"), preset="social_commentary",
                             dialogue_polish=False, sfx_mode="none")
        self.assertFalse(explicit.dialogue_polish)
        self.assertEqual(explicit.sfx_mode, "none")

    def long_transcript(self):
        return Transcript(segments=[Segment(start=i, end=min(i + 12, 678),
            text="What do you think?" if i == 420 else "A neutral piece of documentary narration.")
            for i in range(0, 678, 12)])

    def test_five_chapters_distinct_stems_and_question_closing_holds(self):
        config = RunConfig(script_path=Path("script.txt"), preset="social_commentary")
        sheet = rules_plan(self.long_transcript(), 678, config)
        chapters = chapter_summary(sheet)
        self.assertEqual(len(chapters), 5)
        self.assertTrue(all(90 <= c["end_sec"] - c["start_sec"] <= 150 for c in chapters))
        jobs = [j for j in generation_jobs(sheet) if j[1] == "music"]
        self.assertEqual(len(jobs), 5)
        self.assertEqual(len({j[3] for j in jobs}), 5)
        self.assertTrue(all(j[2] == 60 for j in jobs))
        self.assertTrue(any(s.start_sec <= 431 < s.end_sec for s in sheet.silence_windows))
        self.assertTrue(any(s.end_sec == 678 for s in sheet.silence_windows))

    def test_llm_receives_chapter_contract_without_live_endpoint(self):
        config = RunConfig(script_path=Path("script.txt"), preset="social_commentary", planner="llm")
        baseline = rules_plan(self.long_transcript(), 678, config)
        captured = []
        def response(request, **kwargs):
            captured.append(json.loads(request.data))
            return io.StringIO(json.dumps({"choices": [{"message": {"content": baseline.model_dump_json()}}]}))
        with patch.dict("os.environ", {"DIRECTOR_LLM_URL": "http://local.invalid", "DIRECTOR_LLM_MODEL": "fixture"}), patch("src.director.urlopen", response):
            result = plan(self.long_transcript(), 678, config)
        self.assertEqual(result.planner, "llm")
        self.assertIn("90–150", captured[0]["messages"][0]["content"])
        self.assertEqual(len(chapter_summary(result)), 5)

    def test_documentary_palette_no_environment_and_archive_beats_accent_override(self):
        config = RunConfig(script_path=Path("s.txt"), preset="social_commentary", sfx_mode="full_3_tier",
                           sfx_silence_policy="accents")
        transcript = Transcript(segments=[Segment(start=0, end=8, text="A neutral sentence.")])
        sheet = rules_plan(transcript, 8, config)
        sheet.silence_windows = []
        sheet = protect_archival(sheet, [ArchivalInsertWindow(start_sec=2, end_sec=3, evidence="reviewed news insert")])
        events = []
        for key, at, kind, extra in [("room", 0, "location_change", {"location": "room", "prompt": "birds"}),
            ("cup", 1, "foley", {"prompt": "cup"}), ("stat", 2.5, "data_reveal", {}),
            ("headline", 4, "headline", {}), ("data", 5, "data_reveal", {})]:
            events.append(EditEvent(event_id=key, time_sec=at, kind=kind, confidence=1, intensity=1,
                                    semantic_salience=1, evidence="fixture", **extra))
        result = schedule(EventGraph(events=events), sheet, transcript, config)
        self.assertEqual([x.category for x in result.effects], ["paper_slide", "ui_tick"])
        env = silence_envelope(8 * SR, 0, sheet)
        self.assertEqual(float(env[round(2.2 * SR)]), 0)
        self.assertAlmostEqual(float(env[round(1.925 * SR)]), 0.5, delta=0.01)

    def test_llm_cannot_collapse_all_chapters_into_one_long_loop(self):
        config = RunConfig(script_path=Path("s.txt"), preset="social_commentary", planner="llm")
        baseline = rules_plan(self.long_transcript(), 678, config)
        for cue in baseline.cues:
            cue.chapter_id = "one_long_loop"
        response = json.dumps({"choices": [{"message": {"content": baseline.model_dump_json()}}]})
        with patch.dict("os.environ", {"DIRECTOR_LLM_URL": "http://local.invalid", "DIRECTOR_LLM_MODEL": "fixture"}), \
                patch("src.director.urlopen", return_value=io.StringIO(response)):
            with self.assertRaisesRegex(PipelineError, "schema validation failed"):
                plan(self.long_transcript(), 678, config)

    def test_reference_hysteresis_abstains_on_insufficient_speech(self):
        values = [0.9, 0.2, 0.25, None, 0.85, 0.9]
        scores = [{"start_sec": i * 0.5, "end_sec": i * 0.5 + 2, "cosine": value}
                  for i, value in enumerate(values)]
        windows = mismatch_windows(scores, 5, 0.6, 0.75)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_sec, 0.5)
        self.assertEqual(windows[0].end_sec, 2)
        self.assertFalse(mismatch_windows([dict(s, cosine=None) for s in scores], 5, 0.6, 0.75))


class CommentaryAudio(unittest.TestCase):
    def test_reference_adapter_with_controlled_embeddings_not_real_model(self):
        class ControlledEncoder:
            def __init__(self, **kwargs):
                self.count = 0
            def embed_utterance(self, samples):
                # First call is reference. Source windows at 1.0 and 1.5s mismatch.
                self.count += 1
                return np.array([0., 1.]) if self.count in (4, 5) else np.array([1., 0.])
        module = SimpleNamespace(VoiceEncoder=ControlledEncoder, preprocess_wav=lambda audio, source_sr: audio)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            t = np.arange(6 * SR) / SR
            write_audio(root / "voice.wav", np.column_stack([0.1 * np.sin(2 * np.pi * 260 * t)] * 2))
            config = RunConfig(script_path=Path("fixture.txt"))
            with patch.dict("sys.modules", {"resemblyzer": module}):
                windows, scores = detect_reference_windows(root / "voice.wav", root / "voice.wav", root, config)
            self.assertEqual((windows[0].start_sec, windows[0].end_sec), (1., 2.))
            self.assertTrue(scores)
            self.assertTrue(all("embedding" not in item for item in scores))

    def test_real_vocal_chain_preserves_archival_samples_and_reports_loudness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            t = np.arange(4 * SR) / SR
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t) ** 2
            voice = 0.08 * envelope * np.sin(2 * np.pi * 260 * t) + 0.025 * np.sin(2 * np.pi * 7000 * t)
            source = root / "source.wav"
            write_audio(source, np.column_stack([voice, voice]))
            windows = [ArchivalInsertWindow(start_sec=1, end_sec=2, evidence="controlled insert")]
            output, report = polish_dialogue(source, windows, root)
            original, _ = read_audio(source)
            polished, _ = read_audio(output)
            np.testing.assert_array_equal(polished[SR:2 * SR], original[SR:2 * SR])
            self.assertGreater(float(np.mean(abs(polished[:SR] - original[:SR]))), 1e-3)
            self.assertEqual(len(polished), len(original))
            self.assertTrue(report["archival_samples_preserved"])
            self.assertAlmostEqual(report["narration_only_measurement"]["input_i"], -14, delta=0.6)

    def test_end_to_end_manual_archival_insert_and_doc_accent(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stderr(io.StringIO()):
            root = Path(temp)
            create_demo(root, seconds=6)
            (root / "archive.json").write_text(json.dumps({"windows": [{"start_sec": 1, "end_sec": 2,
                                                                        "evidence": "reviewed news insert"}]}))
            t = np.arange(SR) / SR
            pop = np.exp(-((t - 0.4) / 0.03) ** 2) * np.sin(2 * np.pi * 1500 * t) * 0.15
            write_audio(root / "pop.wav", np.column_stack([pop, pop]))
            event = EditEvent(event_id="evidence", time_sec=3.2, kind="headline", confidence=1,
                              intensity=1, semantic_salience=1, evidence="reviewed headline")
            (root / "events.json").write_text(EventGraph(events=[event]).model_dump_json())
            manifest = json.loads((root / "assets.json").read_text())
            manifest["sfx"]["fx_evidence"] = "pop.wav"
            (root / "assets.json").write_text(json.dumps(manifest))
            config = RunConfig(video_path=root / "raw.mp4", transcript_path=root / "transcript.json",
                mode="final_video", provider="local", local_assets=root / "assets.json", output_dir=root / "out",
                preset="social_commentary", archival_windows_path=root / "archive.json",
                event_graph_path=root / "events.json", pacing_detect_video=False, vocal_policy="off")
            self.assertEqual(execute(config)["status"], "complete")
            branch = root / "out" / "local"
            music, _ = read_audio(branch / "music.wav")
            self.assertEqual(float(np.max(abs(music[SR:2 * SR]))), 0)
            original, _ = read_audio(root / "out" / "dialogue.wav")
            polished, _ = read_audio(root / "out" / "dialogue_polished.wav")
            np.testing.assert_array_equal(polished[SR:2 * SR], original[SR:2 * SR])
            qc = json.loads((branch / "qc_report.json").read_text())
            self.assertEqual(qc["preset"], "social_commentary")
            self.assertTrue(qc["mastering"]["sfx_routing"]["passed"])

    def test_missing_optional_speaker_model_blocks_before_any_provider(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_demo(root, seconds=6)
            config = RunConfig(video_path=root / "raw.mp4", transcript_path=root / "transcript.json",
                mode="assets_and_script", provider="elevenlabs", output_dir=root / "out",
                preset="social_commentary", narrator_reference_path=root / "dialogue.wav")
            with patch.dict("sys.modules", {"resemblyzer": None}), patch("src.pipeline.ElevenLabsProvider",
                    side_effect=AssertionError("provider initialized")):
                with self.assertRaisesRegex(PipelineError, "speaker extra"):
                    execute(config)
