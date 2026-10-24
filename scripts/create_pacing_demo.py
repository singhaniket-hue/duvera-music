"""Offline fixture for all three effects buses and an explicit eye-anchor crop; no API calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.create_demo import create_demo  # noqa: E402
from src.media import SR, write_audio  # noqa: E402
from src.pacing_models import EditEvent, EventGraph  # noqa: E402


def create(root: Path):
    create_demo(root, seconds=6)
    t = np.arange(SR) / SR
    envelope = np.exp(-((t - 0.36) / 0.035) ** 2)
    write_audio(root / "effect.wav", np.column_stack([0.3 * envelope * np.sin(2 * np.pi * 850 * t)] * 2))
    rng = np.random.default_rng(12)
    write_audio(root / "room.wav", rng.normal(0, 0.02, (SR * 3, 2)).astype(np.float32))
    events = [
        dict(event_id="room", time_sec=0, kind="location_change", location="room", prompt="quiet HVAC"),
        dict(event_id="tick", time_sec=1, kind="text_enter"),
        dict(event_id="cup", time_sec=3.5, kind="foley", prompt="soft cup contact"),
        dict(event_id="street", time_sec=4, kind="location_change", location="street", prompt="distant traffic"),
        dict(event_id="crop", time_sec=4, end_sec=5, kind="punch_in", eye_anchor=(0.4, 0.35)),
    ]
    graph = EventGraph(events=[EditEvent(**e, confidence=0.95, intensity=0.95, semantic_salience=0.95,
                                        evidence="Explicit synthetic fixture; not detected real objects/eyes") for e in events])
    (root / "events.json").write_text(graph.model_dump_json(indent=2) + "\n")
    manifest = json.loads((root / "assets.json").read_text())
    manifest["sfx"] = {"fx_room": "room.wav", "fx_street": "room.wav", "fx_tick": "effect.wav", "fx_cup": "effect.wav"}
    (root / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    config = json.loads((root / "config.json").read_text())
    config.update(sfx_mode="full_3_tier", event_graph_path="events.json", pacing_detect_video=False,
                  visual_pacing="render", cue_seconds=3, output_dir="pacing_output")
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("pacing_demo"))
    create(parser.parse_args().dir.resolve())
