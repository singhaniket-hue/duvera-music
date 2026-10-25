"""Frame compositor: real eye measurements, readable documentary cards, H.264 picture track."""
from __future__ import annotations

import math
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .cv_detectors import eye_midpoint, require_cv
from .media import PipelineError, probe
from .visual_pacing import compute_eyeline_anchored_crop


def cubic_bezier(tau, x1=0.22, y1=1., x2=0.36, y2=1.):
    lo, hi = 0., 1.
    tau = float(np.clip(tau, 0, 1))
    for _ in range(30):
        t = (lo + hi) / 2
        x = 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t * t * x2 + t ** 3
        if x < tau:
            lo = t
        else:
            hi = t
    t = (lo + hi) / 2
    return float(3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t * t * y2 + t ** 3)


def spring_step(seconds, zeta=0.78, omega_n=22.):
    if seconds <= 0:
        return 0.
    damped = omega_n * math.sqrt(1 - zeta ** 2)
    return float(np.clip(1 - math.exp(-zeta * omega_n * seconds) *
        (math.cos(damped * seconds) + zeta / math.sqrt(1 - zeta ** 2) * math.sin(damped * seconds)), 0, 1.08))


class SmoothedEyeAnchor:
    """No default face-centre estimate is ever labelled an eye measurement."""
    def __init__(self):
        cv2 = require_cv()
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if self.face.empty() or self.eyes.empty():
            raise PipelineError("OpenCV face/eye cascade data unavailable")
        self.value, self.last_seen, self.last_update = None, -100., -100.

    def update(self, frame, source_time):
        cv2 = require_cv()
        if source_time - self.last_update >= 0.08:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(cv2.resize(frame, (480, max(40, round(h * 480 / w)))), cv2.COLOR_BGR2GRAY)
            found = eye_midpoint(gray, self.face, self.eyes)
            if found:
                if self.value is None or source_time - self.last_seen > 0.4:
                    self.value = np.array(found)
                else:
                    alpha = 1 - math.exp(-(source_time - self.last_update) / 0.35)
                    self.value = (1 - alpha) * self.value + alpha * np.array(found)
                self.last_seen = source_time
            self.last_update = source_time
        return tuple(self.value) if self.value is not None and source_time - self.last_seen <= 0.25 else None


def apply_eyeline_anchored_zoom(frame, ex, ey, zoom):
    if zoom <= 1.0001:
        return frame
    cv2 = require_cv()
    h, w = frame.shape[:2]
    x, y, cw, ch = compute_eyeline_anchored_crop(w, h, (ex, ey), zoom)
    return cv2.resize(frame[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_CUBIC)


@lru_cache(maxsize=64)
def font_path(text, supplied=None):
    try:
        from PIL import features
        from fontTools.ttLib import TTFont
    except ImportError as exc:
        raise PipelineError("Documentary graphics need the edit extra (Pillow/fonttools/regex)") from exc
    hindi = any("\u0900" <= c <= "\u097f" for c in text)
    if hindi and not features.check_feature("raqm"):
        raise PipelineError("Hindi text rendering requires a Pillow build with RAQM shaping")
    candidates = [str(supplied)] if supplied else [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf" if hindi else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc" if hindi else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/Nirmala.ttf" if hindi else "C:/Windows/Fonts/arial.ttf"]
    needed = {ord(c) for c in text if not c.isspace() and c not in "\u200d\u200c"}
    for candidate in candidates:
        if not Path(candidate).is_file():
            continue
        font = TTFont(candidate, fontNumber=0)
        try:
            covered = set((font.getBestCmap() or {}).keys())
        finally:
            font.close()
        if needed <= covered:
            return candidate
    raise PipelineError("No installed font covers this card's text; install Noto fonts or set --documentary-font")


def wrapped(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        proposed = (line + " " + word).strip()
        if draw.textbbox((0, 0), proposed, font=font)[2] > width and line:
            lines.append(line)
            line = word
        else:
            line = proposed
    if line:
        lines.append(line)
    return "\n".join(lines)


def card_layer(event, width, height, supplied_font=None):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise PipelineError("Install the edit extra for documentary graphics") from exc
    font_file = font_path(event.visible_text, str(supplied_font) if supplied_font else None)
    layer = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(layer)
    lower = event.edit_type == "lower_third_citation"
    black = event.edit_type == "black_chapter_card"
    bw = round(width * (0.88 if black else 0.84))
    bh = round(height * (0.52 if black else 0.25 if lower else 0.44))
    x = round((width - bw) / 2) if not lower else round(width * 0.06)
    y = round((height - bh) / 2) if not lower else height - bh - round(height * 0.07)
    padding = max(8, round(width * 0.028))
    top_size = max(14, round(width * (0.047 if black else 0.036 if lower else 0.05)))
    fitted = None
    for size in range(top_size, max(10, round(width * 0.02)) - 1, -1):
        main = ImageFont.truetype(font_file, size)
        sub = ImageFont.truetype(font_file, max(10, round(size * 0.66)))
        title = wrapped(draw, event.headline_text or "", main, bw - 2 * padding)
        citation = wrapped(draw, event.subtext_source or "", sub, bw - 2 * padding)
        box = draw.multiline_textbbox((0, 0), title, font=main, spacing=round(size * 0.2))
        sub_box = draw.multiline_textbbox((0, 0), citation, font=sub, spacing=3) if citation else (0, 0, 0, 0)
        th, sh = box[3] - box[1], sub_box[3] - sub_box[1]
        gap = max(6, round(size * 0.4)) if citation else 0
        if max(box[2] - box[0], sub_box[2] - sub_box[0]) <= bw - 2 * padding and th + sh + gap <= bh - 2 * padding:
            fitted = (main, sub, title, citation, th, sh, gap, box, sub_box)
            break
    if fitted is None:
        raise PipelineError(f"Card {event.event_id} cannot fit readable text at this resolution; shorten/wrap the copy")
    main, sub, title, citation, th, sh, gap, box, sub_box = fitted
    if not black:
        bh = min(bh, max(round(height * 0.16), th + sh + gap + 2 * padding))
        y = round((height - bh) / 2) if not lower else height - bh - round(height * 0.07)
    if not black:
        draw.rounded_rectangle((x, y, x + bw, y + bh), radius=max(6, round(width * 0.014)),
            fill=(19, 23, 30, 245), outline=(218, 176, 89, 255), width=max(1, round(width * 0.002)))
        draw.rectangle((x, y + 8, x + max(3, round(width * 0.005)), y + bh - 8), fill=(218, 176, 89, 255))
    ty = y + (bh - th - sh - gap) // 2
    tx = x + padding if not black else (width - (box[2] - box[0])) // 2
    draw.multiline_text((tx - box[0], ty - box[1]), title, font=main, spacing=round(main.size * 0.2),
                        fill=(248, 245, 237, 255), align="center" if black else "left")
    if citation:
        sx = x + padding if not black else (width - (sub_box[2] - sub_box[0])) // 2
        draw.multiline_text((sx - sub_box[0], ty + th + gap - sub_box[1]), citation, font=sub, spacing=3,
                            fill=(218, 176, 89, 255), align="center" if black else "left")
    return layer, {"event_id": event.event_id, "font": Path(font_file).name, "font_px": main.size,
                   "card_bounds": [x, y, x + bw, y + bh], "wrapped_headline": title}


def render_documentary_overlay(frame, event, time, layer=None, supplied_font=None):
    cv2 = require_cv()
    from PIL import Image
    h, w = frame.shape[:2]
    if layer is None:
        layer, _ = card_layer(event, w, h, supplied_font)
    elapsed, remaining = time - event.start_sec, event.end_sec - time
    enter = cubic_bezier(elapsed / 0.18, 0.16, 1, 0.30, 1)
    leave = cubic_bezier(remaining / 0.12, 0.40, 0, 1, 1)
    alpha = min(enter, leave)
    if event.edit_type == "black_chapter_card":
        background = np.zeros_like(frame)
    elif event.edit_type == "evidence_card":
        dark = (cv2.GaussianBlur(frame, (0, 0), max(2, w * 0.012)) * 0.36).astype(np.uint8)
        background = cv2.addWeighted(dark, alpha, frame, 1 - alpha, 0)
    else:
        background = frame
    shifted = Image.new("RGBA", (w, h))
    shift = round((1 - spring_step(max(0, elapsed))) * min(45, h * 0.08))
    shifted.alpha_composite(layer, (0, shift))
    shifted.putalpha(shifted.getchannel("A").point(lambda value: round(value * alpha)))
    base = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB)).convert("RGBA")
    return cv2.cvtColor(np.asarray(Image.alpha_composite(base, shifted).convert("RGB")), cv2.COLOR_RGB2BGR)


def render_documentary_video_track(source_path, destination, events, mapping, source, supplied_font=None):
    cv2 = require_cv()
    layers, layouts = {}, []
    for event in events:
        if event.is_graphic:
            layers[event.event_id], layout = card_layer(event, source["width"], source["height"], supplied_font)
            layouts.append(layout)
    tracker = SmoothedEyeAnchor() if any(not e.is_graphic and e.eye_anchor is None for e in events) else None
    counts = {e.event_id: {"visible_frames": 0, "zoom_frames": 0, "missing_eye_frames": 0} for e in events}
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise PipelineError("Cannot open source picture track")
    width, height = source["width"], source["height"]
    log_path = destination.with_suffix(".ffmpeg.log")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", source["rate"], "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination)]
    produced, cursor = 0, 0
    with log_path.open("wb") as log:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=log, stdout=subprocess.DEVNULL)
        try:
            for source_index in range(source["frames"]):
                ok, frame = cap.read()
                if not ok:
                    raise PipelineError("Picture decoder ended before the validated source frame count")
                if frame.shape[:2] != (height, width):
                    raise PipelineError("Decoded display orientation/dimensions differ from the source plan")
                while cursor < len(mapping.keep_ranges) and source_index >= mapping.keep_ranges[cursor].source_end_frame:
                    cursor += 1
                if cursor >= len(mapping.keep_ranges) or source_index < mapping.keep_ranges[cursor].source_start_frame:
                    continue
                time = produced / mapping.fps
                active = [e for e in events if e.start_sec <= time < e.end_sec]
                for event in active:
                    counts[event.event_id]["visible_frames"] += 1
                    if event.is_graphic:
                        continue
                    anchor = event.eye_anchor or (tracker.update(frame, source_index / mapping.fps) if tracker else None)
                    if anchor is None:
                        counts[event.event_id]["missing_eye_frames"] += 1
                        continue
                    progress = min(cubic_bezier((time - event.start_sec) / 0.18),
                                   cubic_bezier((event.end_sec - time) / 0.12, 0.40, 0, 1, 1))
                    scale = 1 + (0.20 if event.edit_type == "punch_in_120_emphasis" else 0.14) * progress
                    frame = apply_eyeline_anchored_zoom(frame, anchor[0] * width, anchor[1] * height, scale)
                    counts[event.event_id]["zoom_frames"] += 1
                for event in active:
                    if event.is_graphic:
                        frame = render_documentary_overlay(frame, event, time, layers[event.event_id])
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                produced += 1
            process.stdin.close()
            if process.wait(timeout=1800):
                raise PipelineError("Documentary picture encode failed: " + log_path.read_text(errors="replace")[-2000:])
        except Exception:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.kill()
            process.wait(timeout=30)
            raise
        finally:
            cap.release()
    stream = next(s for s in probe(destination)["streams"] if s["codec_type"] == "video")
    actual = int(stream.get("nb_frames", -1))
    passed = actual == produced == mapping.output_frames
    for event in events:
        info = counts[event.event_id]
        if event.required and (not info["visible_frames"] or (not event.is_graphic and
                              info["zoom_frames"] < info["visible_frames"] * 0.8)):
            passed = False
    if not passed:
        raise PipelineError("Documentary picture QC failed: missing frames or a required eye-anchored event")
    return {"status": "rendered", "passed": passed, "source_frames": source["frames"],
        "output_frames": produced, "events": counts, "layouts": layouts,
        "note": "Haar eye-box geometry is a proxy; automatic zooms without reliable eyes are skipped and counted"}
