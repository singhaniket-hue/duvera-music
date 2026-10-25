"""Prepare the user's reviewed vlog-audit timings and pin an existing MUSIC-ONLY bed.

No media generation, no upload, no narration separation. The base config must point to
raw source video/dialogue with the same edit clock. Audited times are not CV measurements.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cli import PATHS  # noqa: E402
from src.models import RunConfig, SpottingSheet  # noqa: E402
from src.music_continuity import sha256, validate_locked_music  # noqa: E402
from src.pacing_models import EditEvent, EventGraph  # noqa: E402


def breakfast_events() -> EventGraph:
    events = []
    def add(key, time, kind, **extra):
        events.append(EditEvent(event_id=key, time_sec=time, kind=kind, confidence=1,
            intensity=0.95, semantic_salience=0.95, required=True,
            evidence="User audit; black/text boundaries refined from attached 43f2955 frame PTS; same source edit required", **extra))
    for key, time, prompt in [
        ("courtyard", 0, "Quiet morning courtyard air and distant birds"),
        ("auto_outbound", 6, "Soft Bengaluru auto-rickshaw engine and distant traffic, low-pass 4kHz"),
        ("lobby", 16, "Soft luxury hotel lobby ventilation, no close conversations"),
        ("buffet", 22, "Very distant buffet room tone and soft dishes, no close voices"),
        ("room", 39, "Quiet hotel room ventilation"),
        ("auto_return", 47, "Soft auto-rickshaw engine and distant traffic, low-pass 4kHz"),
    ]:
        add(key, time, "location_change", location=key, prompt=prompt, end_sec=68)
    for index, time in enumerate([24.1, 25.3, 27.0, 29.0], 1):
        add(f"plate_{index}", time, "shutter", prompt="Very soft short camera shutter, no voice, no music")
    add("phone_call", 31.2, "foley", prompt="Soft generic video-call connection chime, no speech, no music")
    add("black_reveal", 68, "black_card")
    # Six requested cards, refined to the supplied render's observed frame boundaries.
    # A later 84.2s text change is deliberately left unaccented, keeping the six-tick brief.
    for index, time in enumerate([69.166667, 71, 73.9, 75.8, 79.066667, 81.666667], 1):
        add(f"card_{index}", time, "text_enter")
    return EventGraph(events=events)


def prepare(base_config: Path, music_bed: Path, directory: Path, spotting_sheet: Path | None = None):
    if directory.exists():
        raise ValueError("Recipe destination already exists; choose a fresh directory")
    data = json.loads(base_config.read_text(encoding="utf-8"))
    for key in PATHS:
        if data.get(key):
            path = Path(data[key]).expanduser()
            data[key] = str(path.resolve() if path.is_absolute() else (base_config.parent / path).resolve())
    sheet_path = spotting_sheet or data.get("spotting_sheet_path")
    if not sheet_path:
        for candidate in (music_bed.parent / "spotting_sheet.json", music_bed.parent.parent / "spotting_sheet.json"):
            if candidate.is_file():
                sheet_path = candidate
                break
    if not sheet_path:
        raise ValueError("Supply the original music run's --spotting-sheet to preserve its music/silence decisions")
    sheet_path = Path(sheet_path).resolve()
    sheet = SpottingSheet.model_validate_json(sheet_path.read_text(encoding="utf-8"))
    directory = directory.resolve()
    data.update(mode="final_video", sfx_mode="full_3_tier", sfx_mix_profile="audible",
        sfx_silence_policy="accents", pacing_detect_video=False, event_graph_path=str(directory / "events.json"),
        locked_music_path=str(music_bed.resolve()), locked_music_sha256=sha256(music_bed),
        spotting_sheet_path=str(sheet_path), output_dir=str(directory / "render5"),
        visual_pacing="off", allow_paid_generation=False, vocal_policy="reject")
    config = RunConfig.model_validate(data)
    validate_locked_music(config, sheet.video_duration_sec, sheet)
    if sheet.video_duration_sec <= 82.1:
        raise ValueError("Breakfast recipe requires the original 86-second edit")
    directory.mkdir(parents=True)
    (directory / "events.json").write_text(breakfast_events().model_dump_json(indent=2) + "\n")
    (directory / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    print(str(directory / "config.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--music-bed", required=True, type=Path)
    parser.add_argument("--spotting-sheet", type=Path)
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.base_config.resolve(), args.music_bed.resolve(), args.dir, args.spotting_sheet)
