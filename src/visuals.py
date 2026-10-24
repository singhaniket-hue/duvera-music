"""Streaming luminance boundary detection. Sparse highlights suggest cards, not OCR text."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .media import PipelineError, probe
from .models import RunConfig, VisualEvent


def classify_frames(metrics: list[dict], duration: float, config: RunConfig) -> list[VisualEvent]:
    events, active, preceding = [], [], None
    dt = 1 / config.visual_sample_fps

    def finish(terminal=False):
        if not active:
            return
        start = active[0]["time"]
        end = duration if terminal else min(duration, active[-1]["time"] + dt)
        # Require a material drop from a non-dark scene, not night footage or a startup black frame.
        abrupt = preceding is not None and preceding["mean"] >= 25 and active[0]["mean"] < preceding["mean"] * 0.35
        if end - start >= config.black_min_duration_sec and abrupt:
            bright = float(np.max([m["bright_ratio"] for m in active]))
            events.append(VisualEvent(start_sec=start, end_sec=end,
                                      kind="black_card_candidate" if 0.0003 <= bright <= 0.08 else "dark_boundary",
                                      mean_luma=float(np.mean([m["mean"] for m in active])),
                                      black_pixel_ratio=float(np.mean([m["black_ratio"] for m in active])),
                                      bright_pixel_ratio=bright))

    previous = None
    for frame in metrics:
        dark = frame["mean"] <= config.black_luma_threshold and frame["black_ratio"] >= config.black_pixel_ratio
        if dark:
            if not active:
                preceding = previous
            active.append(frame)
        elif active:
            finish()
            active = []
        previous = frame
    finish(terminal=True)
    return events


def analyze_video(path: Path, duration: float, config: RunConfig) -> list[VisualEvent]:
    if not config.analyze_visuals or not any(s["codec_type"] == "video" for s in probe(path)["streams"]):
        return []
    width, height = 96, 160
    args = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
            "-an", "-vf", f"fps={config.visual_sample_fps},scale={width}:{height},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    # Read bounded frames from stdout; stderr is disk-backed to avoid pipe deadlock.
    metrics = []
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                raw = bytearray()
                while len(raw) < width * height:
                    part = proc.stdout.read(width * height - len(raw))
                    if not part:
                        break
                    raw.extend(part)
                if not raw:
                    break
                if len(raw) != width * height:
                    raise PipelineError("Truncated visual analysis frame")
                frame = np.frombuffer(raw, dtype=np.uint8)
                metrics.append({"time": len(metrics) / config.visual_sample_fps,
                                "mean": float(frame.mean()),
                                "black_ratio": float(np.mean(frame < 20)),
                                "bright_ratio": float(np.mean(frame > 110))})
            if proc.wait(timeout=30):
                errors.seek(0)
                raise PipelineError("Visual analysis failed: " + errors.read(2000).decode(errors="replace"))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
    return classify_frames(metrics, duration, config)
