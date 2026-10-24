"""Perceptual scheduler regressions: real audio/video, no paid generation or semantic claims."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from scripts.create_demo import create_demo
from src.cv_detectors import CardTracker, detect_video, global_motion, stable_anchor
from src.director import rules_plan
from src.dsp_mixer import spectral_duck
from src.event_scheduler import schedule
from src.media import PipelineError, SR, ffmpeg, probe, read_audio, write_audio
from src.models import RunConfig, Segment, Transcript
from src.pacing_models import EditEvent, EventGraph
from src.pipeline import _gate, execute
from src.providers import generation_jobs
from src.providers.elevenlabs import ElevenLabsProvider
from src.sfx_engine import crest_sample, level_effect, placement, shape, window_loudness
from src.visual_pacing import compute_eyeline_anchored_crop, cubic_bezier_y_from_x, validate_render_source


def event(key, time, kind="text_enter", **kwargs):
    return EditEvent(event_id=key, time_sec=time, kind=kind, confidence=0.95,
                     intensity=0.95, semantic_salience=0.95, evidence="Controlled fixture", **kwargs)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.config = RunConfig(script_path=Path("fixture.txt"), sfx_mode="full_3_tier")
        self.transcript = Transcript(segments=[Segment(start=0, end=8, text="A neutral narrated example.")])
        self.sheet = rules_plan(self.transcript, 8, self.config)
        self.sheet.silence_windows = []

    def test_confidence_cooldown_and_ordinary_cuts(self):
        graph = EventGraph(events=[event("cut", 1, "cut"), event("a", 2), event("b", 2.2), event("c", 3)])
        result = schedule(graph, self.sheet, self.transcript, self.config)
        self.assertEqual([e.event_id for e in result.effects], ["a", "c"])
        self.assertIn("ordinary cut", result.decisions[0]["reason"])
        self.assertEqual(result.decisions[2]["reason"], "refractory cooldown")

    def test_protection_is_default_and_explicit_accent_override_excludes_foley(self):
        from src.models import SilenceWindow
        self.sheet.silence_windows = [SilenceWindow(start_sec=2, end_sec=6, reason="joke")]
        graph = EventGraph(events=[event("title", 3, "black_card"), event("cup", 4, "foley", prompt="cup contact")])
        self.assertFalse(schedule(graph, self.sheet, self.transcript, self.config).effects)
        self.config.sfx_silence_policy = "accents"
        result = schedule(graph, self.sheet, self.transcript, self.config)
        self.assertEqual([e.tier for e in result.effects], ["accent"])

    def test_locations_hold_through_angles_and_crossfade_around_cut(self):
        graph = EventGraph(events=[event("room", 0, "location_change", location="room", prompt="HVAC"),
            event("angle", 2, "location_change", location="room", prompt="HVAC"),
            event("street", 4, "location_change", location="street", prompt="distant traffic")])
        result = schedule(graph, self.sheet, self.transcript, self.config)
        self.assertEqual(len(result.effects), 2)
        self.assertAlmostEqual(result.effects[0].end_sec, 4.55)
        self.assertEqual(result.effects[1].j_lead_sec, 0.45)
        self.assertIn("same acoustic location", result.decisions[1]["reason"])
        self.assertTrue(all(0.5 <= j[2] <= 30 for j in generation_jobs(self.sheet, result) if j[1] == "sfx"))

    def test_optional_modes_and_no_invented_semantics(self):
        result = schedule(EventGraph(), self.sheet, self.transcript, self.config)
        self.assertIn("not invented", result.warnings[0])
        self.config.sfx_mode = "none"
        self.assertFalse(schedule(EventGraph(events=[event("tick", 1)]), self.sheet, self.transcript, self.config).effects)

    def test_contract_rejects_unanchored_geometry_and_duplicate_ids(self):
        with self.assertRaises(ValidationError):
            event("eyes", 2, "punch_in", eye_anchor=(1.1, 0.4))
        with self.assertRaises(ValidationError):
            EventGraph(events=[event("same", 1), event("same", 2)])
        with self.assertRaises(ValidationError):
            event("location", 1, "location_change")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            schedule(EventGraph(events=[event("late", 9)]), self.sheet, self.transcript, self.config)

    def test_sfx_budget_counts_before_spend_and_google_requires_imports(self):
        result = schedule(EventGraph(events=[event("a", 1), event("b", 3)]), self.sheet, self.transcript, self.config)
        config = self.config.model_copy(update={"provider": "elevenlabs", "max_requests": 1})
        with self.assertRaisesRegex(PipelineError, "exceeding"):
            _gate(config, self.sheet, result)
        config.provider, config.max_requests = "google_flow", 30
        with self.assertRaisesRegex(PipelineError, "imported assets"):
            _gate(config, self.sheet, result)
        _, payload = ElevenLabsProvider.payload("air", 0.5, "sfx", loop=True)
        self.assertTrue(payload["loop"])


class AudioTimingTests(unittest.TestCase):
    def asset(self):
        t = np.arange(SR) / SR
        envelope = np.exp(-((t - 0.36) / 0.035) ** 2)
        return np.column_stack([0.3 * envelope * np.sin(2 * np.pi * 850 * t)] * 2).astype(np.float32)

    def test_shaping_preserves_input_and_remeasures_landmark(self):
        audio = self.asset()
        before = audio.copy()
        shaped = shape(audio, 0.08, "text_pop")
        np.testing.assert_array_equal(audio, before)
        self.assertEqual(len(shaped), round(0.08 * SR))
        peak = crest_sample(shaped)
        position = placement(1, 8, peak)
        self.assertEqual(position["start_sample"] + peak, SR + 384)

    def test_negative_start_trims_source_without_moving_peak(self):
        shaped = shape(self.asset(), 0.08, "text_pop")
        peak = crest_sample(shaped)
        pos = placement(0, 8, peak)
        self.assertEqual(pos["start_sample"] + peak - pos["source_trim_samples"], 384)

    def test_short_loudness_uses_full_400ms_zero_padded_window(self):
        t = np.arange(round(0.08 * SR)) / SR
        audio = np.column_stack([np.sin(2 * np.pi * 1000 * t)] * 2).astype(np.float32) * 0.05
        short, long = window_loudness(audio[:len(audio) // 2], 0.4), window_loudness(audio, 0.4)
        self.assertAlmostEqual(long - short, 3.01, delta=0.1)
        shaped, report = level_effect(audio, -21, -7)
        self.assertLessEqual(report["reconstructed_peak_estimate_db"], -7)
        self.assertTrue(np.isfinite(shaped).all())

    def test_spectral_duck_protects_speech_and_preserves_stereo_shape(self):
        t = np.arange(SR) / SR
        effect = np.column_stack([np.sin(2 * np.pi * 1700 * t)] * 2).astype(np.float32) * 0.2
        speech = effect * 0.2
        result, report = spectral_duck(effect, speech, "accent")
        self.assertEqual(result.shape, effect.shape)
        self.assertLess(np.mean(result ** 2), np.mean(effect ** 2) * 0.5)
        self.assertLessEqual(max(report["measured_max_band_reduction_db"]), 6.001)
        untouched, _ = spectral_duck(effect, np.zeros_like(speech), "accent")
        np.testing.assert_allclose(untouched[1000:-1000], effect[1000:-1000], atol=1e-5)


class CVTests(unittest.TestCase):
    def test_stable_masks_ignore_jitter_and_pending_changes_must_agree(self):
        tracker = CardTracker()
        picture = np.full((180, 320), 100, np.uint8)
        card_a = np.zeros_like(picture)
        card_a[70:80, 80:180] = 255
        card_b = np.zeros_like(picture)
        card_b[100:110, 150:250] = 255
        tracker.update(picture, 0)
        outputs = []
        for i, frame in enumerate([card_a] * 6 + [np.roll(card_a, 1, axis=1)] * 3 +
                                  [card_b, card_a, card_b, card_b, card_b], 1):
            outputs.extend(tracker.update(frame, i / 30))
        self.assertEqual([x[0] for x in outputs], ["black_card", "text_enter"])
        self.assertAlmostEqual(outputs[-1][1], 12 / 30)
        gradual = CardTracker()
        outputs = []
        for i, value in enumerate(range(100, -1, -5)):
            frame = np.full_like(picture, value)
            frame[70:80, 80:180] = 255
            outputs.extend(gradual.update(frame, i / 30))
        self.assertFalse(outputs)

    def test_ransac_separates_local_motion_from_camera_motion(self):
        import cv2
        rng = np.random.default_rng(4)
        image = (rng.random((180, 320)) * 255).astype(np.uint8)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 12], [0, 1, 0]]), (320, 180))
        camera = global_motion(image, shifted, 1 / 30)
        self.assertGreater(camera["coherence"], 0.65)
        self.assertGreater(camera["coverage"], 0.3)
        self.assertGreater(camera["translation"], 0.8)
        local = image.copy()
        local[60:110, 100:150] = shifted[60:110, 100:150]
        subject = global_motion(image, local, 1 / 30)
        self.assertLess(subject["translation"], 0.1)

    def test_bezier_and_eye_invariance(self):
        crop = compute_eyeline_anchored_crop(1920, 1080, (740, 380), 1.14)
        x, y, w, h = crop
        self.assertAlmostEqual((740 - x) / w, 740 / 1920, delta=1 / w)
        self.assertAlmostEqual((380 - y) / h, 380 / 1080, delta=1 / h)
        values = [cubic_bezier_y_from_x(t) for t in np.linspace(0, 1, 40)]
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], 1)
        samples = [(i / 10, (0.4 + i * 0.0001, 0.35)) for i in range(10)]
        self.assertIsNotNone(stable_anchor(samples, 0, 1))
        self.assertIsNone(stable_anchor([(t, None) for t, _ in samples], 0, 1))
        self.assertIsNone(stable_anchor([(t, (0.4 + t * 0.2, 0.35)) for t, _ in samples], 0, 1))


class PacingIntegration(unittest.TestCase):
    def test_real_three_tier_render_sync_silence_qc_and_zoom(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_demo(root, seconds=6)
            effect = root / "effect.wav"
            write_audio(effect, AudioTimingTests().asset())
            original_hash = hashlib.sha256(effect.read_bytes()).hexdigest()
            events = [event("room", 0, "location_change", location="room", prompt="quiet HVAC"),
                event("tick", 1), event("foley", 3.5, "foley", prompt="soft cup contact"),
                event("street", 4, "location_change", location="street", prompt="distant traffic"),
                event("zoom", 4, "punch_in", end_sec=5, eye_anchor=(0.4, 0.35))]
            graph = root / "events.json"
            graph.write_text(EventGraph(events=events).model_dump_json())
            manifest = json.loads((root / "assets.json").read_text())
            manifest["sfx"] = {"fx_" + e.event_id: "effect.wav" for e in events if e.kind != "punch_in"}
            (root / "assets.json").write_text(json.dumps(manifest))
            config = RunConfig(video_path=root / "raw.mp4", transcript_path=root / "transcript.json",
                mode="final_video", provider="local", local_assets=root / "assets.json", cue_seconds=3,
                output_dir=root / "output", vocal_policy="off", sfx_mode="full_3_tier",
                event_graph_path=graph, pacing_detect_video=False, visual_pacing="render")
            with patch("src.providers.elevenlabs.ElevenLabsProvider.generate", side_effect=AssertionError("paid call")):
                summary = execute(config)
            self.assertEqual(summary["status"], "complete")
            branch = root / "output" / "local"
            report = json.loads((branch / "sfx_qc.json").read_text())
            self.assertTrue(report["passed"])
            for record in report["effects"]:
                if record.get("landmark_error_ms") is not None:
                    self.assertLessEqual(abs(record["landmark_error_ms"]), 5)
            for tier in ("ambience", "foley", "accent", "sfx"):
                audio, rate = read_audio(branch / f"{tier}.wav")
                self.assertEqual(len(audio), 6 * SR)
                self.assertEqual(rate, SR)
                self.assertEqual(np.max(np.abs(audio[round(2.1 * SR):3 * SR])), 0)
            self.assertEqual(hashlib.sha256(effect.read_bytes()).hexdigest(), original_hash)
            qc = json.loads((branch / "qc_report.json").read_text())
            self.assertEqual(qc["mastering"]["visual_pacing"], "rendered")
            info = probe(branch / qc["mastering"]["video"])
            self.assertAlmostEqual(float(info["format"]["duration"]), 6, delta=0.1)
            import cv2
            source = cv2.VideoCapture(str(root / "raw.mp4"))
            output = cv2.VideoCapture(str(branch / qc["mastering"]["video"]))
            try:
                source.set(cv2.CAP_PROP_POS_FRAMES, 135)
                output.set(cv2.CAP_PROP_POS_FRAMES, 135)
                ok_source, before = source.read()
                ok_output, after = output.read()
                self.assertTrue(ok_source and ok_output)
                x, y, w, h = compute_eyeline_anchored_crop(320, 180, (128, 63), 1.14)
                expected = cv2.resize(before[y:y + h, x:x + w], (320, 180))
                error_zoom = np.mean((after.astype(float) - expected) ** 2)
                error_uncropped = np.mean((after.astype(float) - before) ** 2)
                self.assertLess(error_zoom, error_uncropped * 0.5)
            finally:
                source.release()
                output.release()
            # A cached plan is reused, rather than allowing detector nondeterminism to
            # create new request IDs after generation authorization.
            config.pacing_detect_video = False
            with patch("src.pipeline.schedule", side_effect=AssertionError("replanned")):
                self.assertEqual(execute(config)["status"], "complete")

    def test_actual_card_clip_uses_frame_presentation_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cards.mp4"
            ffmpeg("-f", "lavfi", "-i", "color=gray:s=320x180:r=30:d=3",
                "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='gte(t,1)',"
                       "drawbox=x=50:y=60:w=100:h=10:color=white:t=fill:enable='between(t,1,1.9)',"
                       "drawbox=x=180:y=100:w=100:h=10:color=white:t=fill:enable='gte(t,2)'",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", path)
            graph, analysis = detect_video(path, 3)
            cards = [e for e in graph.events if e.kind == "black_card"]
            texts = [e for e in graph.events if e.kind == "text_enter"]
            self.assertEqual(len(cards), 1)
            self.assertAlmostEqual(cards[0].time_sec, 1, delta=1 / 30)
            self.assertEqual(len(texts), 1)
            self.assertAlmostEqual(texts[0].time_sec, 2, delta=1 / 30)
            self.assertIn("presentation", analysis["clock"])

    def test_vfr_visual_render_rejected_but_detection_can_plan_it(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vfr.mp4"
            ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=2", "-vf",
                   "select='not(eq(n,20))'", "-vsync", "vfr", "-c:v", "libx264", path)
            with self.assertRaisesRegex(PipelineError, "constant-frame-rate"):
                validate_render_source(path)
            _, analysis = detect_video(path, 2)
            self.assertEqual(analysis["frames_decoded"], 59)
