"""Actual AV cuts, remapped events, readable graphics and procedural SFX; no paid calls."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from scripts.create_documentary_demo import create
from src.archival import ArchivalInsertWindow
from src.cli import PATHS
from src.documentary_pipeline import (build_time_map, edit_audio, pause_candidates,
    remap_event_graph, remap_transcript, remap_visuals)
from src.documentary_schema import DocumentaryEditPlan, PauseTrimInterval, VisualEditEvent, reading_hold
from src.media import PipelineError, SR, probe, read_audio
from src.models import RunConfig, Segment, Transcript, Word
from src.pipeline import execute
from src.procedural_sfx import procedural_jobs, synthesize
from src.providers import AssetStore
from src.video_editor_engine import (apply_eyeline_anchored_zoom, card_layer, font_path,
    render_documentary_overlay)


def fixture():
    transcript = Transcript(segments=[Segment(start=0, end=2, text="An explanation."),
                                      Segment(start=3.2, end=10, text="Another explanation.")])
    audio = np.full((10 * SR, 2), 0.05, np.float32)
    audio[2 * SR:round(3.2 * SR)] = 0
    edl = DocumentaryEditPlan(source_duration_sec=10,
        pause_trims=[PauseTrimInterval(raw_start_sec=2, raw_end_sec=3.2, keep_pause_ms=200)])
    return transcript, audio, edl


class SharedClock(unittest.TestCase):
    def test_required_event_cannot_disappear_or_collapse_from_out_of_bounds(self):
        from src.pacing_models import EditEvent, EventGraph
        transcript, audio, edl = fixture()
        mapping, _ = build_time_map(edl, transcript, audio, [], {"rate": "30", "frames": 300})
        def graph(time, required=True):
            return EventGraph(events=[EditEvent(event_id="event", time_sec=time, kind="ui_tick",
                required=required, confidence=1, evidence="Reviewed fixture event")])
        with self.assertRaisesRegex(PipelineError, "removed pause"):
            remap_event_graph(graph(2.6), mapping)
        with self.assertRaisesRegex(PipelineError, "exceeds"):
            remap_event_graph(graph(11), mapping)
        self.assertFalse(remap_event_graph(graph(2.6, False), mapping).events)
        self.assertEqual(remap_event_graph(graph(6), mapping).events[0].time_sec, 5)

    def test_one_clock_moves_picture_audio_words_visuals_and_archives(self):
        transcript, audio, edl = fixture()
        transcript.words = [Word(start=0, end=2, word="before"), Word(start=3.2, end=9, word="after")]
        mapping, _ = build_time_map(edl, transcript, audio, [], {"rate": "30", "frames": 300})
        self.assertEqual(mapping.output_frames, 270)
        edited, qc = edit_audio(audio, mapping)
        self.assertEqual(len(edited), 9 * SR)
        self.assertEqual(qc["crossfade_ms"], 20)
        remapped = remap_transcript(transcript, mapping)
        self.assertAlmostEqual(remapped.words[1].start, 2.2)
        self.assertAlmostEqual(mapping.time(6), 5)
        card = VisualEditEvent(start_sec=5, end_sec=8, edit_type="evidence_card", headline_text="15% vs 2%")
        moved = remap_visuals(DocumentaryEditPlan(visual_events=[card]), mapping)[0]
        self.assertEqual((moved.start_sec, moved.end_sec), (4, 7))
        self.assertEqual(mapping.time(10), 9)

    def test_audio_bridge_has_no_extra_duration_loss_or_noise_floor_hole(self):
        transcript, audio, edl = fixture()
        rng = np.random.default_rng(6)
        audio[2 * SR:round(3.2 * SR)] = rng.normal(0, 1e-5, (round(1.2 * SR), 2))
        mapping, _ = build_time_map(edl, transcript, audio, [], {"rate": "30", "frames": 300})
        result, qc = edit_audio(audio, mapping)
        join = mapping.keep_ranges[1].output_start_frame / mapping.fps
        region = result[round((join - 0.01) * SR):round((join + 0.01) * SR)]
        self.assertGreater(np.mean(region ** 2), 1e-12)
        self.assertLess(np.mean(region ** 2), 1e-8)
        self.assertEqual(qc["joins"], 1)
        self.assertEqual(len(result), round(mapping.duration * SR))

    def test_speech_and_archival_regions_cannot_be_trimmed(self):
        transcript, audio, edl = fixture()
        with self.assertRaisesRegex(PipelineError, "archival"):
            build_time_map(edl, transcript, audio, [ArchivalInsertWindow(start_sec=2.5, end_sec=2.8,
                           evidence="protected insert")], {"rate": "30", "frames": 300})
        edl.pause_trims[0] = PauseTrimInterval(raw_start_sec=1, raw_end_sec=3.2, keep_pause_ms=200)
        with self.assertRaisesRegex(PipelineError, "speech"):
            build_time_map(edl, transcript, audio, [], {"rate": "30", "frames": 300})

    def test_dramatic_holds_and_audible_breath_middle_survive_auto_trimmer(self):
        transcript, audio, _ = fixture()
        audio[round(2.3 * SR):round(2.9 * SR)] = 0.006
        trims, _ = pause_candidates(transcript, audio, [])
        self.assertFalse(trims)
        transcript = Transcript(segments=[Segment(start=0, end=2, text="Why does this happen?"),
                                          Segment(start=2.7, end=4, text="Let us consider it.")])
        trims, decisions = pause_candidates(transcript, np.zeros((4 * SR, 2), np.float32), [])
        self.assertFalse(trims)
        self.assertIn("preserved", decisions[0]["reason"])

    def test_fractional_frame_rate_keeps_sample_count_and_event_clock(self):
        transcript, audio, edl = fixture()
        audio = np.pad(audio, ((0, 480), (0, 0)))
        mapping, _ = build_time_map(edl, transcript, audio, [], {"rate": "30000/1001", "frames": 300})
        result, _ = edit_audio(audio, mapping)
        self.assertEqual(len(result), round(mapping.output_frames * float(Fraction(1001, 30000)) * SR))
        self.assertAlmostEqual(mapping.time(8) - mapping.time(7), 1)


class VisualContracts(unittest.TestCase):
    def test_reading_hold_and_invalid_plan_rejection(self):
        with self.assertRaises(ValidationError):
            VisualEditEvent(start_sec=0, end_sec=1, edit_type="evidence_card", headline_text="Too short")
        with self.assertRaises(ValidationError):
            VisualEditEvent(start_sec=0, end_sec=float("nan"), edit_type="punch_in_114")
        with self.assertRaises(ValidationError):
            DocumentaryEditPlan(source_duration_sec=10, visual_events=[
                VisualEditEvent(start_sec=0, end_sec=3, edit_type="evidence_card", headline_text="A short quote"),
                VisualEditEvent(start_sec=2, end_sec=5, edit_type="black_chapter_card", headline_text="A question?")])
        self.assertGreaterEqual(reading_hold("अध्ययन ७५%"), 1.5)

    def test_landscape_and_portrait_text_fit_and_three_styles_render(self):
        for width, height in ((960, 540), (540, 960)):
            base = np.full((height, width, 3), 100, np.uint8)
            for kind in ("evidence_card", "lower_third_citation", "black_chapter_card"):
                event = VisualEditEvent(event_id="card", start_sec=0, end_sec=5, edit_type=kind,
                    headline_text="15% vs 2%", subtext_source="Demonstration only")
                layer, layout = card_layer(event, width, height)
                x0, y0, x1, y1 = layout["card_bounds"]
                self.assertTrue(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height)
                result = render_documentary_overlay(base, event, 2, layer)
                self.assertEqual(result.shape, base.shape)
                self.assertGreater(np.max(abs(result.astype(int) - base)), 50)
                if kind == "black_chapter_card":
                    self.assertEqual(int(result[0, 0].sum()), 0)

    def test_hindi_font_coverage_and_real_shaping(self):
        supplied = os.getenv("DUVERA_TEST_HINDI_FONT")
        try:
            path = font_path("भारत में ७५%", supplied)
        except PipelineError:
            if supplied:
                raise
            self.skipTest("Install Noto Devanagari or set DUVERA_TEST_HINDI_FONT for this visual test")
        event = VisualEditEvent(start_sec=0, end_sec=5, edit_type="black_chapter_card", headline_text="भारत में ७५%")
        layer, layout = card_layer(event, 540, 960, path)
        self.assertGreater(np.asarray(layer)[:, :, 3].sum(), 0)
        self.assertGreaterEqual(layout["font_px"], 14)

    def test_eye_anchor_is_preserved_by_crop_geometry_in_actual_pixels(self):
        frame = np.zeros((540, 960, 3), np.uint8)
        frame[188:194, 477:483] = 255
        result = apply_eyeline_anchored_zoom(frame, 480, 191, 1.20)
        y, x = np.nonzero(result[:, :, 0] > 200)
        self.assertAlmostEqual(float(x.mean()), 480, delta=2)
        self.assertAlmostEqual(float(y.mean()), 191, delta=2)


class DocumentaryIntegration(unittest.TestCase):
    def test_source_music_lock_is_derived_and_hashes_both_clocks(self):
        from src.director import rules_plan
        from src.documentary_pipeline import prepare_documentary
        from src.media import write_audio
        from src.music_continuity import sha256
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            transcript, audio, edl = fixture()
            source = root / "dialogue.wav"
            lock = root / "locked.wav"
            write_audio(source, audio)
            write_audio(lock, np.full_like(audio, 0.1))
            original_bytes = lock.read_bytes()
            config = RunConfig(video_path=root / "dummy.mp4", mode="script_only", output_dir=root,
                documentary_edit_mode="plan", locked_music_path=lock, locked_music_sha256=sha256(lock))
            edl.music_sheet = rules_plan(transcript, 10, config)
            with patch("src.documentary_pipeline.plan_edl", return_value=edl), \
                    patch("src.documentary_pipeline.validate_render_source", return_value={"rate": "30", "frames": 300}):
                result = prepare_documentary(config, transcript, 10, source, [], True)
            self.assertEqual(lock.read_bytes(), original_bytes)
            derived = result["config"].locked_music_path
            self.assertNotEqual(derived, lock)
            self.assertEqual(len(read_audio(derived)[0]), 9 * SR)
            self.assertEqual(result["report"]["original_music_sha256"], sha256(lock))
            self.assertEqual(result["report"]["edited_music_sha256"], sha256(derived))

    def config(self, root):
        path = create(root)
        data = json.loads(path.read_text())
        for key in PATHS:
            if data.get(key):
                data[key] = path.parent / data[key]
        return RunConfig.model_validate(data)

    def test_real_full_editor_av_trim_archival_and_encoded_sfx(self):
        import cv2
        with tempfile.TemporaryDirectory() as temp, redirect_stderr(io.StringIO()):
            root = Path(temp)
            config = self.config(root)
            with patch("src.pipeline.ElevenLabsProvider", side_effect=AssertionError("paid provider initialized")):
                summary = execute(config)
            self.assertEqual(summary["status"], "complete")
            branch = root / "output" / "local"
            qc = json.loads((branch / "qc_report.json").read_text())
            edit = qc["documentary_edit"]
            self.assertEqual(edit["visual_render"]["output_frames"], 450)
            self.assertAlmostEqual(edit["removed_sec"], 1)
            self.assertTrue(qc["mastering"]["visual_qc"]["passed"])
            self.assertTrue(qc["mastering"]["sfx_routing"]["passed"])
            for record in qc["sfx_pacing"]["effects"]:
                self.assertLessEqual(abs(record["landmark_error_ms"]), 5)
            sub = next(r for r in qc["sfx_pacing"]["effects"] if r["category"] == "sub_drop")
            self.assertAlmostEqual(sub["actual_peak_sample"] / SR, 8.01, delta=0.001)
            music, _ = read_audio(branch / "music.wav")
            self.assertEqual(float(np.max(abs(music[8 * SR:round(11.5 * SR)]))), 0)
            self.assertEqual(float(np.max(abs(music[5 * SR:6 * SR]))), 0)
            original, _ = read_audio(root / "output" / "dialogue.wav")
            edited, _ = read_audio(root / "output" / "dialogue_edited.wav")
            polished, _ = read_audio(root / "output" / "dialogue_polished.wav")
            np.testing.assert_array_equal(edited[5 * SR:6 * SR], original[6 * SR:7 * SR])
            np.testing.assert_array_equal(polished[5 * SR:6 * SR], edited[5 * SR:6 * SR])
            final = branch / qc["mastering"]["video"]
            self.assertAlmostEqual(float(probe(final)["format"]["duration"]), 15, delta=0.05)
            cap = cv2.VideoCapture(str(final))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 270)
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertLess(float(frame[:50].mean()), 1)
                self.assertGreater(float(frame.max()), 230)
            finally:
                cap.release()
            ledger = json.loads((branch / "stems" / "generation_ledger.json").read_text())
            self.assertEqual(sum(e["seconds"] for e in ledger.values()), 0)
            self.assertTrue(all(e["mode"] == "procedural" for e in ledger.values()))
            with patch("src.documentary_pipeline.urlopen", side_effect=AssertionError("unexpected replanning")):
                self.assertEqual(execute(config)["status"], "complete")

    def test_one_llm_pass_contains_picture_audio_and_music(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            config.documentary_plan_path, config.planner = None, "llm"
            captured = []
            def response(request, **kwargs):
                payload = json.loads(request.data)
                captured.append(payload)
                baseline = json.loads(payload["messages"][1]["content"])["baseline"]
                return io.StringIO(json.dumps({"choices": [{"message": {"content": json.dumps(baseline)}}]}))
            with patch.dict("os.environ", {"DIRECTOR_LLM_URL": "http://local.invalid", "DIRECTOR_LLM_MODEL": "fixture"}), \
                    patch("src.documentary_pipeline.urlopen", response), patch("src.pipeline.plan", side_effect=AssertionError("second director pass")):
                self.assertEqual(execute(config, plan_only=True)["status"], "planned")
            self.assertEqual(len(captured), 1)
            plan = json.loads((root / "output" / "documentary_edit_plan.json").read_text())
            self.assertTrue(plan["visual_events"])
            self.assertTrue(plan["music_sheet"]["cues"])

    def test_procedural_cache_does_not_spend_provider_request_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = RunConfig(script_path=Path("fixture.txt"), procedural_sfx=True, max_requests=1)
            # Use the same scheduler-approved category path the main pipeline uses.
            from src.event_scheduler import schedule
            from src.director import rules_plan
            from src.pacing_models import EditEvent, EventGraph
            transcript = Transcript(segments=[Segment(start=0, end=5, text="A neutral fixture.")])
            sheet = rules_plan(transcript, 5, config)
            graph = EventGraph(events=[EditEvent(event_id=f"tick_{i}", time_sec=i, kind="data_reveal",
                confidence=1, intensity=1, semantic_salience=1, evidence="fixture") for i in (1, 2, 3)])
            pacing = schedule(graph, sheet, transcript, config.model_copy(update={"sfx_mode": "accents_only"}))
            store = AssetStore(root, config, None, None)
            store.procedural = procedural_jobs(pacing, config)
            for effect in pacing.effects:
                first = store.get(effect.asset_id, effect.prompt, effect.generation_sec, "sfx")
                before = first.read_bytes()
                self.assertEqual(store.get(effect.asset_id, effect.prompt, effect.generation_sec, "sfx").read_bytes(), before)
            self.assertEqual(len(store.ledger), 3)
            for category in ("sub_drop", "paper_slide", "ui_tick", "whoosh"):
                synthesize(category, 0.5, root / "one.wav", "deterministic")
                before = (root / "one.wav").read_bytes()
                synthesize(category, 0.5, root / "two.wav", "deterministic")
                self.assertEqual(before, (root / "two.wav").read_bytes())
