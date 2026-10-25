"""Real-footage compositor using MediaPipe iris landmarks and Chromium HTML overlays."""

from __future__ import annotations

import subprocess
import hashlib
import json
from pathlib import Path
import numpy as np
from .cv_detectors import require_cv
from .media import PipelineError, probe
from .html_overlay_renderer import composite_overlay


def cubic_bezier(tau, x1=0.22, y1=1.0, x2=0.36, y2=1.0):
    lo, hi = 0.0, 1.0
    tau = float(np.clip(tau, 0, 1))
    for _ in range(30):
        t = (lo + hi) / 2
        x = 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t * t * x2 + t**3
        if x < tau:
            lo = t
        else:
            hi = t
    t = (lo + hi) / 2
    return float(3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t * t * y2 + t**3)


class SmoothedEyeAnchor:
    """Refined iris centres from real source frames, never an invented face-centre anchor."""

    def __init__(self):
        try:
            import mediapipe as mp

            self.model = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        except ImportError as exc:
            raise PipelineError("Real eye tracking requires mediapipe==0.10.21 from the edit extra") from exc
        self.value, self.last_seen, self.last_update = None, -100.0, -100.0

    def update(self, frame, source_time):
        cv2 = require_cv()
        if source_time - self.last_update >= 0.075:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(frame, (640, round(h * 640 / w))), cv2.COLOR_BGR2RGB)
            result = self.model.process(rgb)
            if result.multi_face_landmarks:
                marks = result.multi_face_landmarks[0].landmark
                found = np.array([(marks[468].x + marks[473].x) / 2, (marks[468].y + marks[473].y) / 2])
                if np.all((found > 0.05) & (found < 0.95)):
                    alpha = (
                        1
                        if self.value is None or source_time - self.last_seen > 0.3
                        else 1 - np.exp(-(source_time - self.last_update) / 0.15)
                    )
                    self.value = found if self.value is None else (1 - alpha) * self.value + alpha * found
                    self.last_seen = source_time
            self.last_update = source_time
        return tuple(self.value) if self.value is not None and source_time - self.last_seen <= 0.25 else None


def apply_eyeline_anchored_zoom(frame, ex, ey, zoom):
    if zoom <= 1.0001:
        return frame
    cv2 = require_cv()
    h, w = frame.shape[:2]
    matrix = np.array([[zoom, 0, (1 - zoom) * ex], [0, zoom, (1 - zoom) * ey]], dtype=np.float64)
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)


def render_documentary_overlay(frame, event, time, layer=None, supplied_font=None):
    if layer is None:
        raise PipelineError("HTML overlay must be pre-rendered with HTMLOverlayRenderer")
    return composite_overlay(frame, event, time, layer)


def _render_documentary_video_track(
    source_path,
    destination,
    events,
    mapping,
    source,
    supplied_font=None,
    *,
    word_timestamps=(),
    cache_root=None,
    studio_grade=True,
):
    cv2 = require_cv()
    from .html_overlay_renderer import HTMLOverlayRenderer

    cv2.setNumThreads(4)
    layers, layouts = HTMLOverlayRenderer(cache_root or destination.parent / "html_overlays").prepare(
        events, source["width"], source["height"], mapping.fps, word_timestamps
    )
    from .media_cutaway import MediaCutawayRenderer

    media = MediaCutawayRenderer(
        events, source["width"], source["height"], destination.parent / "source_credits"
    )
    from .retake_cutter import splice_switches
    from .studio_grade import StudioCommentaryGrade

    switches = splice_switches(mapping)
    tracker = (
        SmoothedEyeAnchor()
        if studio_grade or switches or any(not e.is_graphic and e.eye_anchor is None for e in events)
        else None
    )
    grade = StudioCommentaryGrade(source["width"], source["height"]) if studio_grade else None
    switch_frames = []
    counts = {e.event_id: {"visible_frames": 0, "zoom_frames": 0, "missing_eye_frames": 0} for e in events}
    anchors = []
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise PipelineError("Cannot open source picture track")
    width, height = source["width"], source["height"]
    log_path = destination.with_suffix(".ffmpeg.log")
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        source["rate"],
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
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
                while (
                    cursor < len(mapping.keep_ranges)
                    and source_index >= mapping.keep_ranges[cursor].source_end_frame
                ):
                    cursor += 1
                if (
                    cursor >= len(mapping.keep_ranges)
                    or source_index < mapping.keep_ranges[cursor].source_start_frame
                ):
                    continue
                time = produced / mapping.fps
                anchor = tracker.update(frame, source_index / mapping.fps) if tracker else None
                if grade:
                    frame = grade.apply(frame, anchor)
                current = next((s for s in reversed(switches) if s["frame"] <= produced), None)
                scale = current["scale"] if current else 1.0
                if current and current["frame"] == produced:
                    if anchor is None:
                        raise PipelineError(
                            "Splice camera switch requires a measured eye anchor on the cut frame"
                        )
                    switch_frames.append({**current, "iris_midpoint_normalized": list(anchor)})
                active = [e for e in events if e.start_sec <= time < e.end_sec]
                for event in active:
                    counts[event.event_id]["visible_frames"] += 1
                    if event.is_graphic or event.edit_type in {"media_cutaway", "context_note"}:
                        continue
                    anchor = event.eye_anchor or anchor
                    if anchor is None:
                        counts[event.event_id]["missing_eye_frames"] += 1
                        continue
                    progress = min(
                        cubic_bezier((time - event.start_sec) / 0.18),
                        cubic_bezier((event.end_sec - time) / 0.12, 0.40, 0, 1, 1),
                    )
                    if event.edit_type == "thesis_creep":
                        scale = 1 + 0.16 * np.clip(
                            (time - event.start_sec) / (event.end_sec - event.start_sec - 1 / mapping.fps),
                            0,
                            1,
                        )
                    else:
                        scale = max(
                            scale,
                            1 + (0.20 if event.edit_type == "punch_in_120_emphasis" else 0.14) * progress,
                        )
                    counts[event.event_id]["zoom_frames"] += 1
                    if produced % 15 == 0:
                        anchors.append(
                            {
                                "event_id": event.event_id,
                                "frame": produced,
                                "time_sec": time,
                                "iris_midpoint_normalized": list(anchor),
                                "zoom": scale,
                                "method": "MediaPipe refined iris 468/473; temporal smoothing",
                                "interpolation": "INTER_LANCZOS4",
                            }
                        )
                if scale > 1 and anchor is not None:
                    frame = apply_eyeline_anchored_zoom(frame, anchor[0] * width, anchor[1] * height, scale)
                for event in active:
                    if event.edit_type in {"media_cutaway", "context_note"}:
                        frame = media.frame(event, time, frame)
                    if event.is_graphic:
                        frame = render_documentary_overlay(frame, event, time, layers[event.event_id])
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                produced += 1
                if produced % 300 == 0:
                    print(f"Real footage rendered: {produced}/{mapping.output_frames} frames", flush=True)
            process.stdin.close()
            if process.wait(timeout=1800):
                raise PipelineError(
                    "Documentary picture encode failed: " + log_path.read_text(errors="replace")[-2000:]
                )
        except Exception:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.kill()
            process.wait(timeout=30)
            raise
        finally:
            cap.release()
            media.close()
            if tracker:
                tracker.model.close()
    stream = next(s for s in probe(destination)["streams"] if s["codec_type"] == "video")
    actual = int(stream.get("nb_frames", -1))
    passed = actual == produced == mapping.output_frames
    for event in events:
        info = counts[event.event_id]
        if event.required and (
            not info["visible_frames"]
            or (
                not event.is_graphic
                and event.edit_type not in {"media_cutaway", "context_note"}
                and info["zoom_frames"] < info["visible_frames"] * 0.8
            )
        ):
            passed = False
    if not passed:
        raise PipelineError("Documentary picture QC failed: missing frames or a required eye-anchored event")
    return {
        "status": "rendered",
        "passed": passed,
        "source_frames": source["frames"],
        "output_frames": produced,
        "events": counts,
        "layouts": layouts,
        "eye_anchor_samples": anchors,
        "splice_switches": switch_frames,
        "studio_grade": studio_grade,
        "note": "MediaPipe FaceMesh refined iris centres; smoothed real measurements, no dummy anchors; missing detections are counted",
    }


def render_documentary_video_track(
    source_path,
    destination,
    events,
    mapping,
    source,
    supplied_font=None,
    *,
    word_timestamps=(),
    cache_root=None,
    studio_grade=True,
):
    """Resume only a verified real render with the same source, plan, code and word clock."""
    source_path, destination = Path(source_path), Path(destination)
    source_stat = source_path.stat()
    from .html_overlay_renderer import VERSION

    signature = hashlib.sha256(
        json.dumps(
            {
                "source": [str(source_path.resolve()), source_stat.st_size, source_stat.st_mtime_ns],
                "events": [e.model_dump(mode="json") for e in events],
                "mapping": mapping.model_dump(mode="json"),
                "words": list(word_timestamps),
                "html_version": VERSION,
                "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "grade_sha256": hashlib.sha256(
                    Path(__file__).with_name("studio_grade.py").read_bytes()
                ).hexdigest(),
                "media_sha256": hashlib.sha256(
                    Path(__file__).with_name("media_cutaway.py").read_bytes()
                ).hexdigest(),
                "studio_grade": studio_grade,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    marker = destination.with_suffix(".render.json")
    if marker.exists() and destination.exists():
        cached = json.loads(marker.read_text())
        if (
            cached["signature"] == signature
            and cached["video_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
        ):
            stream = next(s for s in probe(destination)["streams"] if s["codec_type"] == "video")
            if int(stream["nb_frames"]) == mapping.output_frames:
                return {**cached["report"], "verified_render_cache_hit": True}
    report = _render_documentary_video_track(
        source_path,
        destination,
        events,
        mapping,
        source,
        supplied_font,
        word_timestamps=word_timestamps,
        cache_root=cache_root,
        studio_grade=studio_grade,
    )
    marker.write_text(
        json.dumps(
            {
                "signature": signature,
                "video_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "report": report,
            },
            indent=2,
        )
    )
    return report
