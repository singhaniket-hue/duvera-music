"""Prepare a documentary run using real supplied footage; never generate dummy AV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.media import PipelineError  # noqa: E402
from src.models import RunConfig  # noqa: E402


def create(root: Path, video: Path | None = None, transcript: Path | None = None, plan: Path | None = None):
    if not video or not transcript or not video.is_file() or not transcript.is_file():
        raise PipelineError("Real source --video and timestamped --transcript files are required; synthetic demos are disabled")
    root.mkdir(parents=True, exist_ok=False)
    config = RunConfig(preset="social_commentary", video_path=video.resolve(), transcript_path=transcript.resolve(),
        documentary_plan_path=plan.resolve() if plan else None, output_dir=root.resolve() / "render",
        provider="elevenlabs", mode="final_video", procedural_sfx=False, vocal_policy="reject")
    path = root / "config.json"
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--documentary-plan", type=Path)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    print(create(args.dir, args.video, args.transcript, args.documentary_plan))
