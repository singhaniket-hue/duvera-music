"""Eye-anchored crops with actual Bézier inversion; speech and source duration are untouched."""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from .media import PipelineError, probe, run
from .pacing_models import PacingPlan


def cubic_bezier_y_from_x(tau: float) -> float:
    tau = float(np.clip(tau, 0, 1))
    # Bisection is monotonic and converges even near a small endpoint derivative.
    lo, hi = 0., 1.
    for _ in range(32):
        t = (lo + hi) / 2
        x = 3 * (1 - t) ** 2 * t * 0.22 + 3 * (1 - t) * t * t * 0.36 + t ** 3
        if x < tau:
            lo = t
        else:
            hi = t
    t = (lo + hi) / 2
    return float(3 * (1 - t) ** 2 * t + 3 * (1 - t) * t * t + t ** 3)


def compute_eyeline_anchored_crop(width, height, eye_midpoint_xy, zoom_factor=1.14):
    ex, ey = eye_midpoint_xy
    if not (width > 0 and height > 0 and 0 <= ex <= width and 0 <= ey <= height and zoom_factor >= 1):
        raise ValueError("Invalid crop geometry")
    cw, ch = round(width / zoom_factor), round(height / zoom_factor)
    x = round(ex - ex / width * cw)
    y = round(ey - ey / height * ch)
    return max(0, min(width - cw, x)), max(0, min(height - ch, y)), cw, ch


def validate_render_source(path: Path) -> dict:
    streams = [s for s in probe(path)["streams"] if s["codec_type"] == "video"]
    if not streams:
        raise PipelineError("Visual pacing needs a video stream")
    video = streams[0]
    try:
        rate = Fraction(video.get("avg_frame_rate", "0/1"))
    except (ValueError, ZeroDivisionError) as exc:
        raise PipelineError("Video frame rate unavailable") from exc
    if rate <= 0:
        raise PipelineError("Video frame rate unavailable")
    raw = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", path]))
    times = [float(f["best_effort_timestamp_time"]) for f in raw["frames"]
             if "best_effort_timestamp_time" in f]
    if len(times) < 2 or any(abs((t - times[0]) - i / float(rate)) > 0.002 for i, t in enumerate(times)):
        raise PipelineError("Visual rendering currently requires constant-frame-rate input; VFR can still be analyzed/planned")
    if abs(times[0]) > 0.002 or any(float(s.get("start_time", 0)) > 0.002 for s in streams):
        raise PipelineError("Visual render source must start at timestamp zero")
    rotation = float(video.get("tags", {}).get("rotate", 0))
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
    if rotation % 90:
        raise PipelineError("Normalize non-right-angle rotation before visual rendering")
    width, height = video["width"], video["height"]
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return {"width": width, "height": height, "rate": str(rate), "frames": len(times)}


def ffmpeg_filter(plan: PacingPlan, source: dict) -> str:
    rate = float(Fraction(source["rate"]))
    z, ux, uy = "1", "0.5", "0.5"
    for item in reversed(plan.visuals):
        start, end = math.ceil(item.start_sec * rate - 1e-6), math.ceil(item.end_sec * rate - 1e-6) - 1
        steps = max(1, round(item.animation_sec * rate))
        zoom = f"{item.zoom_to:.8f}"
        for step in reversed(range(steps)):
            value = 1 + (item.zoom_to - 1) * cubic_bezier_y_from_x(step / steps)
            zoom = f"if(eq(on,{start + step}),{value:.8f},{zoom})"
        condition = f"between(on,{start},{end})"
        z = f"if({condition},{zoom},{z})"
        ux = f"if({condition},{item.eye_anchor[0]:.8f},{ux})"
        uy = f"if({condition},{item.eye_anchor[1]:.8f},{uy})"
    return (f"zoompan=z='{z}':x='(iw-iw/zoom)*({ux})':y='(ih-ih/zoom)*({uy})':"
            f"d=1:s={source['width']}x{source['height']}:fps={source['rate']},pad=ceil(iw/2)*2:ceil(ih/2)*2")
