"""Generate deterministic OFFLINE test media. Tones simulate speech cadence; no real voice or AI generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.director import rules_plan
from src.media import SR, ffmpeg, write_audio
from src.models import RunConfig, Segment, Transcript


def create_demo(root: Path, seconds=12) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    t = np.arange(round(seconds * SR)) / SR
    env = (np.sin(2 * np.pi * 2 * t) > -0.3) * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t) ** 2)
    voice = (0.12 * np.sin(2 * np.pi * 170 * t) + 0.05 * np.sin(2 * np.pi * 950 * t)) * env
    write_audio(root / "dialogue.wav", np.column_stack([voice, voice]))
    beat = np.exp(-np.mod(t, 0.6) * 25)
    music = (0.14 * np.sin(2 * np.pi * 110 * t) + 0.1 * np.sin(2 * np.pi * 720 * t)) * (0.3 + beat)
    write_audio(root / "music.wav", np.column_stack([music, np.roll(music, 30)]))
    rng = np.random.default_rng(42)
    effect = rng.normal(0, 0.06, size=(2 * SR, 2)).astype(np.float32)
    effect *= np.linspace(0, 1, len(effect))[:, None]
    write_audio(root / "whoosh.wav", effect)
    segments = [
        Segment(start=0, end=seconds / 3, text="You learned to do everything yourself."),
        Segment(start=seconds / 3, end=seconds * 2 / 3, text="The truth is, asking for help can feel unfamiliar."),
        Segment(start=seconds * 2 / 3, end=seconds, text="Start with one safe person and one small request."),
    ]
    transcript = Transcript(segments=segments, timing_source="supplied")
    (root / "transcript.json").write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    (root / "script.txt").write_text("\n".join(s.text for s in segments), encoding="utf-8")
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=30:duration={seconds}",
           "-i", root / "dialogue.wav", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", root / "raw.mp4")
    config = RunConfig(video_path=root / "raw.mp4", transcript_path=root / "transcript.json",
                       mode="final_video", provider="local", local_assets=root / "assets.json",
                       cue_seconds=max(3, seconds / 3 + 0.1), output_dir=root / "output", vocal_policy="off")
    sheet = rules_plan(transcript, seconds, config)
    # Fixed supplied downbeat offsets make offline entry behavior deterministic.
    assets = {"music": {c.cue_id: "music.wav" for c in sheet.cues},
              "sfx": {c.cue_id: "whoosh.wav" for c in sheet.cues},
              "downbeats": {c.cue_id: 0 for c in sheet.cues}}
    (root / "assets.json").write_text(json.dumps(assets, indent=2))
    config_data = config.model_dump(mode="json")
    for key in ("video_path", "transcript_path", "local_assets", "output_dir"):
        config_data[key] = Path(config_data[key]).name
    path = root / "config.json"
    path.write_text(json.dumps(config_data, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("demo_media"))
    args = parser.parse_args()
    print(create_demo(args.dir.resolve()))
