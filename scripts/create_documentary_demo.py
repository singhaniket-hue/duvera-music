"""Offline 16s engineering fixture: real AV cut, reviewed eye anchor, three card styles and SFX."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.director import rules_plan  # noqa: E402
from src.documentary_schema import DocumentaryEditPlan, PauseTrimInterval, VisualEditEvent  # noqa: E402
from src.media import SR, ffmpeg, write_audio  # noqa: E402
from src.models import RunConfig, Segment, Transcript  # noqa: E402


def create(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    seconds = 16
    t = np.arange(seconds * SR) / SR
    rng = np.random.default_rng(15)
    room = rng.normal(0, 0.00004, len(t))
    envelope = 0.3 + 0.7 * (np.sin(2 * np.pi * 2.5 * t) > 0)
    voice = envelope * (0.12 * np.sin(2 * np.pi * 170 * t) + 0.04 * np.sin(2 * np.pi * 950 * t)) + room
    pause = (t >= 3.8) & (t < 5)
    voice[pause] = room[pause]
    archive = (t >= 6) & (t < 7)
    voice[archive] = 0.1 * np.sin(2 * np.pi * 330 * t[archive]) + room[archive]
    write_audio(root / "voice.wav", np.column_stack([voice, voice]))
    music = (0.08 * np.sin(2 * np.pi * 110 * t) + 0.04 * np.sin(2 * np.pi * 500 * t)) * (0.3 + np.exp(-np.mod(t, 0.6) * 18))
    write_audio(root / "music.wav", np.column_stack([music, np.roll(music, 30)]))
    picture = ("drawgrid=w=120:h=120:t=1:c=white@0.05,"
        "drawbox=x=330:y=80:w=300:h=400:c=0x111a24:t=fill,"
        "drawbox=x=425:y=186:w=16:h=10:c=white:t=fill,drawbox=x=519:y=186:w=16:h=10:c=white:t=fill,"
        "drawtext=text='OFFLINE ENGINEERING DEMO':x=250:y=60:fontsize=24:fontcolor=white,"
        "drawtext=text='Synthetic audio':x=380:y=440:fontsize=20:fontcolor=white")
    ffmpeg("-f", "lavfi", "-i", f"color=c=0x263342:s=960x540:r=30:d={seconds}",
        "-i", root / "voice.wav", "-vf", picture, "-c:v", "libx264", "-crf", 18,
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", root / "raw.mp4")
    transcript = Transcript(segments=[
        Segment(start=0, end=3.8, text="A comparison reports 15% versus 2%."),
        Segment(start=5, end=8.8, text="The researcher explains the finding."),
        Segment(start=9, end=12.5, text="Why do people follow the crowd?"),
        Segment(start=13, end=16, text="This is a demonstration, not a verified claim.")])
    (root / "transcript.json").write_text(transcript.model_dump_json(indent=2))
    config = RunConfig(video_path=root / "raw.mp4", transcript_path=root / "transcript.json",
        mode="final_video", preset="social_commentary", provider="local", local_assets=root / "assets.json",
        output_dir=root / "output", documentary_plan_path=root / "edit_plan.json",
        archival_windows_path=root / "archive.json", vocal_policy="off")
    sheet = rules_plan(transcript, seconds, config)
    events = [
        VisualEditEvent(event_id="anchor114", start_sec=0, end_sec=3.7, edit_type="punch_in_114", eye_anchor=(0.5, 191 / 540), required=True),
        VisualEditEvent(event_id="comparison", start_sec=0.6, end_sec=3.7, edit_type="evidence_card", headline_text="15% vs 2%", subtext_source="Demo data", sfx_trigger="ui_data_tick", required=True),
        VisualEditEvent(event_id="anchor120", start_sec=5.2, end_sec=8.8, edit_type="punch_in_120_emphasis", eye_anchor=(0.5, 191 / 540), required=True),
        VisualEditEvent(event_id="citation", start_sec=5.6, end_sec=8.8, edit_type="lower_third_citation", headline_text="Research citation", subtext_source="Demonstration only", sfx_trigger="paper_slide_pop", required=True),
        VisualEditEvent(event_id="question", start_sec=9, end_sec=12.5, edit_type="black_chapter_card", headline_text="Why do we follow?", sfx_trigger="sub_drop", required=True),
    ]
    edl = DocumentaryEditPlan(source_duration_sec=seconds, music_sheet=sheet, visual_events=events,
        pause_trims=[PauseTrimInterval(raw_start_sec=3.8, raw_end_sec=5, keep_pause_ms=200)],
        warnings=["Engineering fixture: synthetic tones and explicitly reviewed geometry; no real face/voice quality claim"])
    (root / "edit_plan.json").write_text(edl.model_dump_json(indent=2))
    (root / "archive.json").write_text(json.dumps({"windows": [{"start_sec": 6, "end_sec": 7,
        "evidence": "Controlled external-audio placeholder in this demo"}]}))
    (root / "assets.json").write_text(json.dumps({"music": {cue.cue_id: "music.wav" for cue in sheet.cues}}))
    data = config.model_dump(mode="json")
    for key, value in config:
        if isinstance(value, Path):
            data[key] = value.name
    (root / "config.json").write_text(json.dumps(data, indent=2))
    return root / "config.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("documentary_demo"))
    print(create(parser.parse_args().dir.resolve()))
