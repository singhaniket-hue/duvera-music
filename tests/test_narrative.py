"""Narrative and rhythm regressions use invented text and synthetic media only."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import find_peaks

from scripts.create_demo import create_demo
from src.director import plan, rules_plan
from src.dsp import duck, transient_exit
from src.media import SR, ffmpeg, read_audio
from src.models import RunConfig, Segment, SpottingSheet, Transcript, VisualEvent
from src.narrative import fully_silent, narrative_events
from src.pipeline import execute
from src.providers import generation_jobs
from src.rhythm import fit_music, plan_loop
from src.visuals import classify_frames


def story() -> Transcript:
    return Transcript(segments=[
        Segment(start=0, end=43, text="A bright cheerful morning at a cafe."),
        Segment(start=43, end=55, text="Par pata nahi kyun it felt different."),
        Segment(start=55, end=68, text="Wapas aate samay main soch mein pad gaya."),
        Segment(start=68, end=74, text="A quiet moment at home."),
        Segment(start=74, end=82, text="Aur finally realize kiya the funny twist."),
        Segment(start=82, end=86, text="Agar aap bhi meri tarah hain try a smaller adventure."),
    ])


def card(start=68, end=86):
    return VisualEvent(start_sec=start, end_sec=end, kind="black_card_candidate",
                       mean_luma=3, black_pixel_ratio=0.98, bright_pixel_ratio=0.01)


class NarrativeRules(unittest.TestCase):
    def config(self):
        return RunConfig(script_path=Path("fiction.txt"), language="hinglish", cue_seconds=120)

    def test_three_acts_protect_punchline_and_return_on_still_black_cta(self):
        sheet = rules_plan(story(), 86, self.config(), visuals=[card()])
        self.assertTrue(fully_silent(68, 82, sheet.silence_windows))
        self.assertFalse(fully_silent(82.1, 86, sheet.silence_windows))
        pensive = next(c for c in sheet.cues if c.start_sec == 55)
        self.assertEqual(pensive.lowpass_hz, 4500)
        contrast = next(c for c in sheet.cues if c.start_sec == 43)
        self.assertEqual(contrast.lowpass_hz, 6500)
        exit_cue = next(c for c in sheet.cues if c.end_sec == 68)
        self.assertEqual(exit_cue.exit_type, "transient_stop")
        self.assertEqual(exit_cue.exit_tail_ms, 200)
        resumed = next(c for c in sheet.cues if c.start_sec == 82)
        self.assertEqual(resumed.lowpass_hz, 9500)
        silent_ids = {c.cue_id for c in sheet.cues if fully_silent(c.start_sec, c.end_sec, sheet.silence_windows)}
        self.assertFalse(silent_ids & {c.cue_id for c, _, _, _ in generation_jobs(sheet)})

    def test_contrast_word_alone_does_not_mute(self):
        t = Transcript(segments=[Segment(start=0, end=4, text="A calm morning."),
                                 Segment(start=4, end=8, text="Lekin I prefer tea.")])
        sheet = rules_plan(t, 8, self.config())
        self.assertEqual(sheet.silence_windows, [])
        self.assertEqual(next(c for c in sheet.cues if c.start_sec == 4).lowpass_hz, 6500)

    def test_devanagari_pivots(self):
        t = Transcript(segments=[Segment(start=0, end=3, text="सुबह अच्छी थी"),
                                 Segment(start=3, end=6, text="मैं सोच में पड़ गया"),
                                 Segment(start=6, end=10, text="पेट खराब हो गया"),
                                 Segment(start=10, end=14, text="अगर आप भी ऐसा सोचते हैं")])
        self.assertEqual([e["kind"] for e in narrative_events(t)], ["reflective", "twist", "resume"])
        sheet = rules_plan(t, 14, self.config())
        self.assertTrue(fully_silent(6, 10, sheet.silence_windows))

    def test_word_anchors_locate_two_turns_within_one_asr_segment(self):
        t = Transcript(segments=[Segment(start=0, end=12, text="A morning then pet kharab ho gaya agar aap bhi")],
                       words=[{"start": 4, "end": 4.3, "word": "pet"},
                              {"start": 4.3, "end": 4.8, "word": "kharab"},
                              {"start": 4.8, "end": 5.4, "word": "ho"},
                              {"start": 5.4, "end": 6, "word": "gaya"},
                              {"start": 10, "end": 10.3, "word": "agar"},
                              {"start": 10.3, "end": 10.6, "word": "aap"},
                              {"start": 10.6, "end": 11, "word": "bhi"}])
        events = narrative_events(t)
        self.assertEqual([(e["kind"], e["start"]) for e in events], [("twist", 4), ("resume", 10)])
        sheet = rules_plan(t, 12, self.config())
        self.assertTrue(fully_silent(4, 10, sheet.silence_windows))
        self.assertFalse(fully_silent(10, 12, sheet.silence_windows))

    def test_llm_cannot_omit_observed_black_boundary(self):
        config = self.config()
        config.planner = "llm"
        baseline = rules_plan(story(), 86, config)
        plain = baseline.cues[0].model_copy(update={"start_sec": 0, "end_sec": 86})
        raw = SpottingSheet(video_duration_sec=86, timing_source="supplied", cues=[plain])
        response = {"choices": [{"message": {"content": raw.model_dump_json()}}]}
        with patch.dict("os.environ", {"DIRECTOR_LLM_URL": "http://local.invalid", "DIRECTOR_LLM_MODEL": "test"}):
            with patch("src.director.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
                result = plan(story(), 86, config, visuals=[card()])
        self.assertTrue(fully_silent(68, 82, result.silence_windows))
        self.assertEqual(next(c for c in result.cues if c.start_sec == 55).lowpass_hz, 4500)

    def test_reflective_lpf_changes_actual_spectrum(self):
        t = np.arange(SR) / SR
        tone = 0.1 * np.sin(2 * np.pi * 8000 * t) + 0.1 * np.sin(2 * np.pi * 400 * t)
        audio = np.column_stack([tone] * 2)
        cue = rules_plan(story(), 86, self.config()).cues[0]
        bright, _ = duck(audio, None, cue)
        cue.lowpass_hz = 4500
        dark, _ = duck(audio, None, cue)
        # Color is a high/low spectral ratio, independent of the loudness controller's makeup gain.
        def ratio(signal):
            spectrum = abs(np.fft.rfft(signal[SR // 4:, 0])) ** 2
            freq = np.fft.rfftfreq(len(signal) - SR // 4, 1 / SR)
            return spectrum[(freq > 7000) & (freq < 9000)].sum() / spectrum[(freq > 200) & (freq < 800)].sum()
        self.assertLess(ratio(dark), ratio(bright) * 0.2)


class RhythmAndVisuals(unittest.TestCase):
    def test_beat_quantized_repeat_never_loses_crossfade_time(self):
        audio = np.zeros((round(4.3 * SR), 2), np.float32)
        beats = np.arange(0.1, 4.11, 0.5)
        for beat in beats:
            index = round(beat * SR)
            audio[index:index + 30] = 0.3
        region = plan_loop(audio, {"beats": beats.tolist(), "downbeat_verified": False}, entry=0.1)
        self.assertTrue(region["beat_quantized"])
        self.assertFalse(region["bar_phase_verified"])
        self.assertAlmostEqual(region["period_sec"], 4)
        output = fit_music(audio, 11.2, region=region)
        peaks, _ = find_peaks(output[:, 0], height=0.1, distance=round(0.4 * SR))
        np.testing.assert_array_equal(np.diff(peaks), np.full(len(peaks) - 1, round(0.5 * SR)))
        self.assertEqual(len(output), round(11.2 * SR))

    def test_phase_continues_across_adjacent_same_source_cues(self):
        t = np.arange(5 * SR) / SR
        audio = np.column_stack([np.sin(2 * np.pi * 310 * t) * 0.1] * 2)
        region = plan_loop(audio, {}, explicit={"start_sec": 0.1, "end_sec": 4.1})
        full = fit_music(audio, 8, region=region)
        second = fit_music(audio, 2, region=region, phase_sec=3.2)
        margin = round(0.03 * SR)
        np.testing.assert_allclose(second[margin:-margin],
                                   full[round(3.2 * SR) + margin:round(5.2 * SR) - margin], atol=1e-6)

    def test_missing_grid_and_invalid_loop_are_explicit(self):
        audio = np.ones((SR, 2), np.float32) * 0.1
        self.assertFalse(plan_loop(audio, {"beats": [0.2, 0.6]})["beat_quantized"])
        with self.assertRaises(ValueError):
            plan_loop(audio, {}, explicit={"start_sec": 0.9, "end_sec": 1.1})

    def test_transient_tail_stays_before_boundary(self):
        audio = np.zeros((3 * SR, 2), np.float32)
        index = round(2.25 * SR)
        audio[index:index + round(0.09 * SR)] = np.exp(-np.arange(round(0.09 * SR)) / 1200)[:, None] * 0.2
        output, record = transient_exit(audio, 200)
        self.assertEqual(len(output), len(audio))
        self.assertEqual(record["tail_ms"], 200)
        self.assertTrue(np.any(output[round(2.35 * SR):round(2.6 * SR)]))
        self.assertEqual(abs(output[-500:]).max(), 0)

    def test_black_detection_ignores_flash_startup_and_dim_night(self):
        config = RunConfig(script_path=Path("fiction.txt"))
        def frame(time, mean, ratio=0.98, bright=0.01):
            return {"time": time, "mean": mean, "black_ratio": ratio, "bright_ratio": bright}
        good = [frame(0, 100, 0.2), *[frame(x / 4, 3) for x in range(1, 9)]]
        events = classify_frames(good, 2.29, config)
        self.assertEqual(events[0].kind, "black_card_candidate")
        self.assertEqual(events[0].start_sec, 0.25)
        self.assertEqual(events[0].end_sec, 2.29)
        self.assertEqual(classify_frames(good[:2] + [frame(0.5, 100, 0.1)], 1, config), [])
        self.assertEqual(classify_frames(good[1:], 3, config), [])
        good[0] = frame(0, 15)
        self.assertEqual(classify_frames(good, 3, config), [])


class VisualMixIntegration(unittest.TestCase):
    def test_real_black_cards_mute_stems_and_keep_narration(self):
        with tempfile.TemporaryDirectory(prefix="duvera narrative ") as directory:
            root = Path(directory)
            create_demo(root)
            path = root / "story.mp4"
            ffmpeg("-i", root / "dialogue.wav", "-filter_complex",
                   "testsrc2=size=320x180:rate=30:duration=6[day];"
                   "color=black:size=320x180:rate=30:duration=6,"
                   "drawbox=x=100:y=80:w=120:h=8:color=white:t=fill[card];"
                   "[day][card]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "0:a",
                   "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", path)
            transcript = Transcript(segments=[
                Segment(start=0, end=3, text="A bright morning at a cafe."),
                Segment(start=3, end=5, text="I wondered why it felt different."),
                Segment(start=5, end=6, text="A quiet ride home."),
                Segment(start=6, end=8, text="A moment at home."),
                Segment(start=8, end=10, text="But then I realized the joke."),
                Segment(start=10, end=12, text="If you are like me try one small step."),
            ])
            tp = root / "story.json"
            tp.write_text(transcript.model_dump_json())
            config = RunConfig(video_path=path, transcript_path=tp, output_dir=root / "plan",
                               mode="script_only", cue_seconds=120, vocal_policy="off")
            execute(config)
            sheet = SpottingSheet.model_validate_json((config.output_dir / "spotting_sheet.json").read_text())
            self.assertAlmostEqual(sheet.visual_events[0].start_sec, 6, delta=0.25)
            self.assertTrue(fully_silent(6, 10, sheet.silence_windows))
            manifest = root / "story_assets.json"
            jobs = list(generation_jobs(sheet))
            manifest.write_text(json.dumps({
                "music": {cue.cue_id: "music.wav" for cue, kind, _, _ in jobs if kind == "music"},
                "sfx": {cue.cue_id: "whoosh.wav" for cue, kind, _, _ in jobs if kind == "sfx"},
                "loops": {cue.cue_id: {"start_sec": 0, "end_sec": 4.8}
                          for cue, kind, _, _ in jobs if kind == "music"},
            }))
            config.mode, config.provider, config.local_assets = "final_video", "local", manifest
            config.output_dir = root / "render"
            self.assertEqual(execute(config)["status"], "complete")
            branch = config.output_dir / "local"
            music, _ = read_audio(branch / "music.wav")
            sfx, _ = read_audio(branch / "sfx.wav")
            self.assertEqual(len(music), 12 * SR)
            self.assertEqual(abs(music[6 * SR:10 * SR]).max(), 0)
            self.assertEqual(abs(sfx[6 * SR:10 * SR]).max(), 0)
            self.assertGreater(abs(music[round(10.2 * SR):11 * SR]).max(), 0)
            qc = json.loads((branch / "qc_report.json").read_text())
            reflective = next(c for c in qc["cues"] if c.get("lowpass_hz") == 4500)
            self.assertAlmostEqual(reflective["loop_phase_sec"], 3)
            # Final audio keeps the original voice during the music silence.
            from src.media import extract_audio
            extract_audio(branch / qc["mastering"]["video"], root / "encoded.wav")
            encoded, _ = read_audio(root / "encoded.wav")
            self.assertGreater(np.mean(encoded[7 * SR:8 * SR] ** 2), 1e-5)


if __name__ == "__main__":
    unittest.main()
