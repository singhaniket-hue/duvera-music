"""Local CV evidence, not OCR/VLM semantics. Every frame uses decoded presentation time."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .media import PipelineError, run
from .pacing_models import EditEvent, EventGraph


def require_cv():
    try:
        import cv2
    except ImportError as exc:
        raise PipelineError("Pacing video detection needs the cv extra; install .[cv] or use --no-pacing-detection") from exc
    return cv2


class CardTracker:
    """Pending masks must agree with EACH OTHER before replacing committed text state."""
    def __init__(self):
        self.black_count = 0
        self.seen_picture = False
        self.entered = False
        self.committed = None
        self.pending = None
        self.pending_count = 0
        self.pending_time = 0.0
        self.black_start = 0.0
        self.previous_luma = 0.0
        self.abrupt = False

    @staticmethod
    def iou(a, b):
        union = np.count_nonzero(a | b)
        return np.count_nonzero(a & b) / max(1, union)

    def update(self, gray: np.ndarray, time: float) -> list[tuple[str, float]]:
        cv2 = require_cv()
        black = float(np.median(gray)) < 12 and np.mean(gray < 24) > 0.90
        luma = float(np.mean(gray))
        previous_luma, self.previous_luma = self.previous_luma, luma
        mask = cv2.dilate((gray > 180).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        sparse_text = 0.0015 < np.mean(mask) < 0.12
        found = []
        if not black:
            self.seen_picture = True
            self.black_count, self.entered = 0, False
            self.committed, self.pending, self.pending_count = None, None, 0
            return found
        if not self.black_count:
            self.black_start = time
            self.abrupt = previous_luma - luma > 12
        self.black_count += 1
        if self.black_count >= 2 and not self.entered and sparse_text and self.seen_picture and self.abrupt:
            self.entered = True
            found.append(("black_card", self.black_start))
        if not sparse_text:
            self.pending, self.pending_count = None, 0
            return found
        if self.committed is not None and self.iou(mask, self.committed) >= 0.3:
            self.pending, self.pending_count = None, 0
            return found
        if self.pending is None or self.iou(mask, self.pending) < 0.7:
            self.pending, self.pending_count, self.pending_time = mask, 1, time
        else:
            self.pending_count += 1
        if self.pending_count >= 3:
            if self.committed is not None and self.entered:
                found.append(("text_enter", self.pending_time))
            self.committed, self.pending, self.pending_count = mask, None, 0
        return found


def global_motion(previous, current, dt: float) -> dict:
    cv2 = require_cv()
    empty = {"translation": 0., "rotation": 0., "zoom": 0., "coherence": 0., "coverage": 0.}
    p0 = cv2.goodFeaturesToTrack(previous, maxCorners=300, qualityLevel=0.02, minDistance=8)
    if p0 is None or len(p0) < 12:
        return empty
    p1, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, p0, None)
    if p1 is None or status is None:
        return empty
    a, b = p0[status.ravel() == 1], p1[status.ravel() == 1]
    if len(a) < 12:
        return empty
    matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2)
    if matrix is None or inliers is None:
        return empty
    h, w = current.shape
    points = a[inliers.ravel() == 1].reshape(-1, 2)
    if len(points) < 8:
        return empty
    span = np.ptp(points, axis=0)
    coverage = float(span[0] * span[1] / (w * h))
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    return {"translation": math.hypot(matrix[0, 2] / w, matrix[1, 2] / h) / dt,
        "rotation": abs(math.atan2(matrix[1, 0], matrix[0, 0])) / dt,
        "zoom": abs(math.log(max(scale, 1e-6))) / dt,
        "coherence": float(inliers.mean()), "coverage": coverage}


def eye_midpoint(gray, face_detector, eye_detector):
    """Conservative Haar geometry; an eye-box midpoint is not a pupil/landmark model."""
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45))
    if len(faces) != 1:
        return None
    x, y, w, h = faces[0]
    upper = gray[y:y + round(h * 0.65), x:x + w]
    eyes = eye_detector.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
    if len(eyes) != 2:
        return None
    centres = sorted([(ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in eyes])
    if centres[1][0] - centres[0][0] < w * 0.25 or abs(centres[1][1] - centres[0][1]) > h * 0.15:
        return None
    return ((x + (centres[0][0] + centres[1][0]) / 2) / gray.shape[1],
            (y + (centres[0][1] + centres[1][1]) / 2) / gray.shape[0])


def stable_anchor(samples, start: float, end: float):
    interval = [(t, value) for t, value in samples if start <= t < end]
    valid = [(t, value) for t, value in interval if value is not None]
    if len(valid) < 3 or len(valid) / max(1, len(interval)) < 0.8 or valid[-1][0] - valid[0][0] < 0.25:
        return None
    points = np.array([value for _, value in valid])
    centre = np.median(points, axis=0)
    if np.max(np.linalg.norm(points - centre, axis=1)) > 0.025:
        return None
    return tuple(float(x) for x in centre)


def detect_video(path: Path, duration: float, punch_candidates=None) -> tuple[EventGraph, dict]:
    cv2 = require_cv()
    # No frame_idx / fps estimate: use the presentation clock, including VFR inputs.
    metadata = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", path]))
    timestamps = [float(x["best_effort_timestamp_time"]) for x in metadata["frames"]
                  if "best_effort_timestamp_time" in x]
    if not timestamps:
        return EventGraph(), {"status": "no_video_frames"}
    timestamps = [t - timestamps[0] for t in timestamps]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise PipelineError("Pacing detection needs strictly increasing video presentation timestamps")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise PipelineError("OpenCV could not decode video for pacing")
    tracker, previous, prev_hist, prev_edge = CardTracker(), None, None, None
    events, motion_records = [], []
    motion_sequence, last_motion = [], -10.
    last_cut, previous_time = -10., 0.
    eye_samples, last_eye_sample = [], -1.
    face_detector = eye_detector = None
    if punch_candidates:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if face_detector.empty() or eye_detector.empty():
            raise PipelineError("OpenCV eye/face cascade data unavailable")

    def finish_motion():
        nonlocal last_motion
        if len(motion_sequence) >= 2:
            apex = max(motion_sequence, key=lambda m: m["translation"] + m["rotation"] * 0.3 + m["zoom"])
            if apex["time_sec"] - last_motion > 0.8:
                kind = "zoom" if apex["zoom"] > apex["translation"] else "whip"
                events.append(EditEvent(event_id=f"cv_motion_{round(apex['time_sec'] * 1000000)}",
                    time_sec=apex["time_sec"], kind=kind, confidence=apex["coherence"],
                    intensity=apex["salience"], semantic_salience=0.8,
                    evidence="Peak coherent RANSAC motion within multi-frame candidate; no semantic cause inferred"))
                last_motion = apex["time_sec"]
        motion_sequence.clear()
    decoded = 0
    try:
        for index, time in enumerate(timestamps):
            ok, frame = cap.read()
            if not ok:
                raise PipelineError("OpenCV/ffprobe frame count mismatch; timing cannot be trusted")
            decoded += 1
            if time >= duration:
                break
            height, width = frame.shape[:2]
            small = cv2.resize(frame, (320, max(32, round(height * 320 / width))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if punch_candidates and time - last_eye_sample >= 0.08 and any(
                    p["start"] <= time < min(p["end"], p["start"] + 2.5) for p in punch_candidates):
                face_gray = cv2.cvtColor(cv2.resize(frame, (640, round(height * 640 / width))), cv2.COLOR_BGR2GRAY)
                eye_samples.append((time, eye_midpoint(face_gray, face_detector, eye_detector)))
                last_eye_sample = time
            for kind, at in tracker.update(gray, time):
                events.append(EditEvent(event_id=f"cv_{kind}_{index}", time_sec=at, kind=kind,
                    confidence=0.9, intensity=0.9, semantic_salience=0.9 if kind == "black_card" else 0.85,
                    evidence="Frame-PTS black/sparse-highlight mask transition, stable across frames; NOT OCR"))
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist /= max(float(hist.sum()), 1)
            edge = cv2.Canny(gray, 80, 160) > 0
            if previous is not None:
                hist_delta = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
                luma_delta = abs(float(gray.mean()) - float(previous.mean())) / 255
                edge_delta = np.count_nonzero(edge ^ prev_edge) / max(1, np.count_nonzero(edge | prev_edge))
                # Two independent image modalities must change. Cut alone emits NO sound.
                is_cut = ((hist_delta > 0.55 and edge_delta > 0.65) or
                          (luma_delta > 0.22 and edge_delta > 0.65))
                if is_cut and time - last_cut >= 0.15:
                    events.append(EditEvent(event_id=f"cv_cut_{index}", time_sec=time, kind="cut",
                        confidence=0.8, intensity=float(min(1, hist_delta)), evidence="Histogram/luma plus edge change"))
                    last_cut = time
                motion = global_motion(previous, gray, time - previous_time) if not is_cut else {
                    "translation": 0., "rotation": 0., "zoom": 0., "coherence": 0., "coverage": 0.}
                history = motion_records[-max(15, round(2 / (time - previous_time))):]
                components = []
                for key in ("translation", "rotation", "zoom"):
                    values = [m[key] for m in history] or [0]
                    p50, p95 = np.percentile(values, [50, 95])
                    components.append(float(np.clip((motion[key] - p50) / max(p95 - p50, 0.01), 0, 1)))
                salience = max(0.55 * components[0] + 0.25 * components[1] + 0.20 * components[2],
                               0.8 * max(components))
                fast = motion["translation"] > 0.6 or motion["rotation"] > 1.5 or motion["zoom"] > 0.6
                candidate = (not is_cut and fast and salience > 0.7 and motion["coherence"] > 0.65
                             and motion["coverage"] > 0.3 and time - last_cut > 0.1)
                if candidate:
                    motion_sequence.append({"time_sec": time, **motion, "salience": salience})
                else:
                    finish_motion()
                motion_records.append({"time_sec": time, **motion, "salience": salience})
            previous, prev_hist, prev_edge, previous_time = gray, hist, edge, time
    finally:
        cap.release()
    finish_motion()
    for index, candidate in enumerate(punch_candidates or []):
        start = candidate["start"]
        end = min(candidate["end"], start + 2.5, duration)
        end = min(end, next((e.time_sec for e in events if e.kind == "cut" and e.time_sec > start + 0.1), end))
        anchor = stable_anchor(eye_samples, start, end)
        if anchor:
            events.append(EditEvent(event_id=f"cv_punch_{index}", time_sec=start, end_sec=end,
                kind="punch_in", confidence=0.8, semantic_salience=0.8, eye_anchor=anchor,
                evidence=f"Transcript {candidate['kind']} ({candidate['anchor']} timing); stable Haar eye-box midpoint; review"))
    return EventGraph(events=events), {"status": "analyzed", "frames_decoded": decoded,
        "clock": "ffprobe_best_effort_presentation_timestamps_relative_to_first_frame",
        "text_detector": "stable_bright_masks_NOT_OCR", "semantic_location_detector": "not_available_supply_event_graph",
        "eye_tracker": "conservative_Haar_eye_box_midpoints_stability_gate" if punch_candidates else "not_requested",
        "motion": motion_records}
