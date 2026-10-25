"""Pure shared-clock contracts; real-footage graphics/SFX checks live in test_cinema_upgrade."""
from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

from src.archival import ArchivalInsertWindow
from src.documentary_pipeline import (build_time_map, edit_audio, pause_candidates,
    remap_event_graph, remap_transcript, remap_visuals)
from src.documentary_schema import DocumentaryEditPlan, PauseTrimInterval, VisualEditEvent
from src.media import PipelineError, SR
from src.models import Segment, Transcript, Word


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
