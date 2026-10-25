"""Regression fixtures for the Render #4 audit: bus routing, level loss, missed cards and intros."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.create_demo import create_demo
from scripts.prepare_breakfast_v5 import breakfast_events
from src.cv_detectors import CardTracker
from src.director import rules_plan
from src.dsp_mixer import render_effects
from src.event_scheduler import schedule
from src.media import PipelineError, SR, extract_audio, ffmpeg, read_audio, write_audio
from src.mix_qc import verify_sfx_routing
from src.models import RunConfig, Segment, SilenceWindow, Transcript
from src.music_continuity import audible_entry, music_gap_qc, sha256
from src.pacing_models import EditEvent, EventGraph
from src.pipeline import execute
from src.sfx_engine import sample_peak_db


def tone(seconds, frequency=2400):
    t = np.arange(round(seconds * SR)) / SR
    envelope = np.exp(-((t - 0.36) / 0.045) ** 2)
    return np.column_stack([0.15 * envelope * np.sin(2 * np.pi * frequency * t)] * 2).astype(np.float32)


class AudibilityRegressions(unittest.TestCase):
    def fixture_sheet(self, seconds=86.266667):
        config = RunConfig(script_path=Path("fixture.txt"), sfx_mode="full_3_tier", sfx_mix_profile="audible",
                           sfx_silence_policy="accents")
        transcript = Transcript(segments=[Segment(start=0, end=seconds, text="A neutral narrated fixture.")])
        sheet = rules_plan(transcript, seconds, config)
        sheet.silence_windows = [SilenceWindow(start_sec=68, end_sec=seconds, reason="Protected music punchline")]
        return config, transcript, sheet

    def test_recipe_has_required_exact_events_j_leads_and_no_silent_skips(self):
        config, transcript, sheet = self.fixture_sheet()
        graph = breakfast_events()
        plan = schedule(graph, sheet, transcript, config)
        self.assertFalse(plan.blocking_issues)
        sub = [x for x in plan.effects if x.category == "sub_drop"]
        pops = [x for x in plan.effects if x.category == "text_pop"]
        self.assertEqual(len(sub), 1)
        self.assertEqual([x.visual_time_sec for x in pops], [69.166667, 71, 73.9, 75.8, 79.066667, 81.666667])
        self.assertEqual(sub[0].target_peak_dbfs, -3.5)
        self.assertTrue(all(x.target_peak_dbfs == -8 for x in pops))
        ambience = [x for x in plan.effects if x.tier == "ambience"]
        self.assertIn(5.55, [x.visual_time_sec - x.j_lead_sec for x in ambience])
        self.assertIn(46.55, [x.visual_time_sec - x.j_lead_sec for x in ambience])
        config.sfx_silence_policy = "protect"
        self.assertTrue(schedule(graph, sheet, transcript, config).blocking_issues)

    def test_actual_86_second_bus_keeps_sub_and_six_ticks_at_requested_peaks(self):
        config, transcript, sheet = self.fixture_sheet()
        graph = EventGraph(events=[e for e in breakfast_events().events if e.kind in {"black_card", "text_enter"}])
        config.sfx_mode = "accents_only"
        plan = schedule(graph, sheet, transcript, config)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_audio(root / "sfx.wav", np.zeros((round(sheet.video_duration_sec * SR), 2), np.float32))
            for item in plan.effects:
                write_audio(root / "stems" / f"{item.asset_id}_sfx.wav", tone(0.8, 60 if item.category == "sub_drop" else 2400))
            report = render_effects(plan, sheet, None, None, root, config)
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["effects"]), 7)
            for item, record in zip(plan.effects, report["effects"]):
                self.assertAlmostEqual(record["routed_sample_peak_dbfs"], item.target_peak_dbfs, delta=0.3)
                self.assertLessEqual(abs(record["landmark_error_ms"]), 5)
                self.assertAlmostEqual(record["actual_peak_sample"] / SR,
                    item.visual_time_sec + item.av_offset_ms / 1000, delta=0.005)
            audio, _ = read_audio(root / "sfx_stem.wav")
            self.assertGreater(sample_peak_db(audio[68 * SR:round(68.6 * SR)]), -4)
            self.assertEqual(sha256(root / "sfx.wav"), sha256(root / "sfx_stem.wav"))

    def test_tiny_vertical_text_and_same_position_word_changes_are_detected(self):
        picture = np.full((568, 320), 120, np.uint8)
        first = np.zeros_like(picture)
        first[270:274, 150:180] = 255  # 0.066%, below the old 0.15% threshold.
        second = first.copy()
        second[270:274, 150:165] = 0
        tracker = CardTracker()
        tracker.update(picture, 0)
        found = []
        for index, frame in enumerate([first] * 5 + [second] * 5):
            found += tracker.update(frame, 1 + index / 30)
        self.assertEqual([kind for kind, _ in found], ["black_card", "text_enter"])
        noise = CardTracker()
        noise.update(picture, 0)
        sparse = np.zeros_like(picture)
        sparse[::60, ::60] = 255
        self.assertFalse(sum((noise.update(sparse, 1 + i / 30) for i in range(6)), []))

    def test_long_source_intro_is_trimmed_and_unplanned_timeline_gap_fails(self):
        audio = np.pad(tone(1), ((5 * SR, 0), (0, 0)))
        self.assertGreater(audible_entry(audio), 5)
        _, _, sheet = self.fixture_sheet(seconds=86.266667)
        music = np.full((round(sheet.video_duration_sec * SR), 2), 0.01, np.float32)
        music[round(39.5 * SR):44 * SR] = 0
        music[68 * SR:] = 0
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "music.wav"
            write_audio(path, music)
            qc = music_gap_qc(path, sheet, 0.75)
            self.assertFalse(qc["passed"])
            self.assertEqual(len(qc["unexpected_gaps"]), 1)
            self.assertAlmostEqual(qc["unexpected_gaps"][0]["start_sec"], 39.5)

    def test_real_ffmpeg_mixing_detects_normalize_one_attenuation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = tone(1)
            write_audio(root / "sfx.wav", audio)
            write_audio(root / "ref.wav", np.zeros_like(audio))
            ffmpeg("-i", root / "ref.wav", "-c:a", "pcm_f32le", root / "reference.wav")
            for normalize in (0, 1):
                ffmpeg("-i", root / "ref.wav", "-i", root / "sfx.wav", "-filter_complex",
                    f"amix=inputs=2:normalize={normalize}", "-c:a", "pcm_f32le", root / "mixed.wav")
                qc = verify_sfx_routing(root / "mixed.wav", root / "reference.wav", root / "sfx.wav")
                self.assertEqual(qc["passed"], normalize == 0)

    def test_enabled_but_empty_plan_blocks_before_provider_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "script.txt").write_text("A neutral line with several ordinary words.")
            config = RunConfig(script_path=root / "script.txt", mode="assets_and_script", provider="elevenlabs",
                               sfx_mode="accents_only", output_dir=root / "out")
            with patch("src.pipeline.ElevenLabsProvider", side_effect=AssertionError("provider initialized")):
                with self.assertRaisesRegex(PipelineError, "no effects accepted"):
                    execute(config)


class LockedAndEncodedIntegration(unittest.TestCase):
    def test_lock_preserves_music_skips_generation_and_sfx_survives_aac(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stderr(io.StringIO()):
            root = Path(temp)
            create_demo(root, seconds=6)
            transcript = Transcript.model_validate_json((root / "transcript.json").read_text())
            config = RunConfig(video_path=root / "raw.mp4", transcript_path=root / "transcript.json",
                mode="final_video", provider="local", local_assets=root / "assets.json", cue_seconds=3,
                output_dir=root / "out", vocal_policy="off", pacing_detect_video=False,
                sfx_mode="accents_only", sfx_mix_profile="audible", sfx_silence_policy="accents")
            sheet = rules_plan(transcript, 6, config)
            sheet.silence_windows = [SilenceWindow(start_sec=3, end_sec=6, reason="fixture black card")]
            (root / "sheet.json").write_text(sheet.model_dump_json())
            config.spotting_sheet_path = root / "sheet.json"
            t = np.arange(6 * SR) / SR
            bed = np.column_stack([0.03 * np.sin(2 * np.pi * 110 * t)] * 2)
            bed[3 * SR:] = 0
            write_audio(root / "locked.wav", bed)
            config.locked_music_path, config.locked_music_sha256 = root / "locked.wav", sha256(root / "locked.wav")
            tick = EditEvent(event_id="tick", time_sec=3.5, kind="text_enter", confidence=1,
                intensity=1, semantic_salience=1, required=True, evidence="controlled fixture")
            (root / "events.json").write_text(EventGraph(events=[tick]).model_dump_json())
            config.event_graph_path = root / "events.json"
            write_audio(root / "tick.wav", tone(0.8))
            (root / "assets.json").write_text(json.dumps({"sfx": {"fx_tick": "tick.wav"}}))
            with patch("src.pipeline.ElevenLabsProvider", side_effect=AssertionError("music regenerated")):
                result = execute(config)
            self.assertEqual(result["status"], "complete")
            branch = root / "out" / "local"
            self.assertEqual(sha256(branch / "music.wav"), config.locked_music_sha256)
            jobs = json.loads((root / "out" / "generation_plan.json").read_text())["jobs_per_generated_branch"]
            self.assertEqual([j["kind"] for j in jobs], ["sfx"])
            qc = json.loads((branch / "qc_report.json").read_text())
            self.assertTrue(qc["mastering"]["sfx_routing"]["passed"])
            self.assertTrue(qc["mastering"]["sfx_routing"]["nonzero"])
            extract_audio(branch / qc["mastering"]["video"], root / "encoded.wav")
            audio, _ = read_audio(root / "encoded.wav")
            def energy_at(start):
                data = audio[round(start * SR):round((start + 0.12) * SR)].mean(axis=1)
                spectrum = np.abs(np.fft.rfft(data * np.hanning(len(data)))) ** 2
                freq = np.fft.rfftfreq(len(data), 1 / SR)
                return float(spectrum[(freq > 2300) & (freq < 2500)].sum())
            self.assertGreater(energy_at(3.48), energy_at(3.2) * 20)
            # Tampered/stale bed is rejected before any provider, even if generation is authorized.
            config.output_dir = root / "tampered"
            config.locked_music_sha256 = "0" * 64
            with patch("src.pipeline.ElevenLabsProvider", side_effect=AssertionError("provider initialized")):
                with self.assertRaisesRegex(PipelineError, "SHA-256 mismatch"):
                    execute(config)
