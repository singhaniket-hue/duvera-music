"""FFmpeg boundary: argument lists only, timeouts, actionable errors."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 48000


class PipelineError(RuntimeError):
    pass


def run(args: list, timeout: float = 1800) -> str:
    try:
        result = subprocess.run([str(a) for a in args], capture_output=True, text=True,
                                timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise PipelineError(f"Tool unavailable or timed out: {args[0]}") from exc
    if result.returncode:
        raise PipelineError(f"{args[0]} failed: {result.stderr[-2500:]}")
    return result.stdout + result.stderr


def ffmpeg(*args):
    return run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-y", *args])


def probe(path: Path) -> dict:
    return json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))


def extract_audio(path: Path, output: Path, duration: float | None = None):
    trim = ["-t", duration] if duration is not None else []
    ffmpeg("-i", path, *trim, "-vn", "-ac", "2", "-ar", SR, "-c:a", "pcm_s16le", output)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise PipelineError("Internal WAV must be PCM 16-bit")
        rate, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return data.reshape(-1, channels), rate


def write_audio(path: Path, data: np.ndarray, rate: int = SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if not np.isfinite(data).all():
        raise PipelineError("Non-finite audio samples")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(data, -1, 1 - 1 / 32768) * 32768).astype("<i2").tobytes())


def loudness(path: Path, target: float = -15, peak: float = -1.5) -> dict:
    raw = ffmpeg("-i", path, "-af", f"loudnorm=I={target}:TP={peak}:LRA=7:print_format=json",
                 "-f", "null", "-")
    try:
        stats = json.loads(raw[raw.rfind("{"):raw.rfind("}") + 1])
        return {k: float(v) if math.isfinite(float(v)) else None for k, v in stats.items()
                if k != "normalization_type"}
    except (ValueError, KeyError) as exc:
        raise PipelineError("Could not parse FFmpeg loudness measurement") from exc


def normalize(source: Path, output: Path, target: float, peak: float):
    stats = loudness(source, target, peak)
    keys = ["input_i", "input_lra", "input_tp", "input_thresh", "target_offset"]
    if any(stats.get(k) is None for k in keys):
        raise PipelineError("Cannot normalize silent or unmeasurable mix")
    filt = (f"loudnorm=I={target}:TP={peak}:LRA=7:measured_I={stats['input_i']}:"
            f"measured_LRA={stats['input_lra']}:measured_TP={stats['input_tp']}:"
            f"measured_thresh={stats['input_thresh']}:offset={stats['target_offset']}:linear=true")
    ffmpeg("-i", source, "-af", filt, "-ar", SR, "-c:a", "pcm_s24le", output)


def require_ffmpeg():
    if not all(shutil.which(x) for x in ("ffmpeg", "ffprobe")):
        raise PipelineError("Install FFmpeg and ffprobe, then rerun --check-env")
