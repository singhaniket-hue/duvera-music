"""Phase-preserving loop regions. A beat grid is not proof of meter/downbeats."""
from __future__ import annotations

import numpy as np

from .media import SR


def plan_loop(audio: np.ndarray, analysis: dict, entry: float = 0,
              explicit: dict | None = None, beats_per_bar: int = 4) -> dict:
    duration = len(audio) / SR
    if explicit:
        start, end = float(explicit["start_sec"]), float(explicit["end_sec"])
        if not 0 <= start < end <= duration or end - start < 0.1:
            raise ValueError("Supplied loop region must lie inside the source audio")
        return {"start_sec": start, "end_sec": end, "method": explicit.get('method','user_supplied_loop'),
                "beat_quantized": explicit.get('beat_quantized',True),
                "bar_phase_verified": explicit.get('bar_phase_verified','user_supplied'),
                "period_sec": end - start}
    beats = np.asarray(analysis.get("beats", []), dtype=float)
    beats = beats[np.isfinite(beats)]
    beats = np.unique(beats[(beats >= max(0, entry - 0.08)) & (beats < duration - 0.02)])
    if len(beats) >= beats_per_bar + 1:
        intervals = np.diff(beats)
        median = float(np.median(intervals))
        stable = 0.28 <= median <= 1.5 and np.max(abs(intervals - median)) <= median * 0.12
        count = ((len(beats) - 1) // beats_per_bar) * beats_per_bar
        if stable and count >= beats_per_bar:
            start, end = float(beats[0]), float(beats[count])
            verified = bool(analysis.get("downbeat_verified")) and abs(start - entry) < 0.08
            return {"start_sec": start, "end_sec": end, "period_sec": end - start,
                    "method": "stable_beat_grid", "beat_quantized": True,
                    "beats_per_bar_assumed": beats_per_bar,
                    "bar_phase_verified": verified,
                    "warning": None if verified else "Meter/bar phase assumed, not a verified downbeat"}
    start = min(max(0, entry), max(0, duration - 1 / SR))
    return {"start_sec": start, "end_sec": duration, "period_sec": duration - start,
            "method": "sample_period_fallback", "beat_quantized": False, "bar_phase_verified": False,
            "warning": "No stable beat grid; fixed sample-period crossfade used. Supply verified loop endpoints."}


def fit_music(audio: np.ndarray, seconds: float, onset: float = 0, *,
              region: dict | None = None, phase_sec: float = 0, crossfade_ms: float = 40) -> np.ndarray:
    """Exact target length and repeat period: crossfades NEVER subtract time from a loop."""
    if not len(audio):
        raise ValueError("Empty music asset")
    target = round(seconds * SR)
    region = region or {"start_sec": onset, "end_sec": len(audio) / SR}
    start = min(max(0, round(region["start_sec"] * SR)), len(audio) - 1)
    end = min(len(audio), round(region["end_sec"] * SR))
    if end <= start:
        raise ValueError("Empty loop region")
    period = end - start
    phase = (np.arange(target, dtype=np.int64) + round(phase_sec * SR)) % period
    result = audio[start + phase].astype(np.float32, copy=True)
    fade = min(round(crossfade_ms / 1000 * SR), period // 4)
    if fade and target + round(phase_sec * SR) > period:
        if start >= fade:
            # Preroll matches the same beat phase before the next downbeat; keep the onset intact.
            mask = phase >= period - fade
            pos = phase[mask] - (period - fade)
            ramp = (pos / fade).astype(np.float32)[:, None]
            result[mask] = result[mask] * (1 - ramp) + audio[start - fade + pos] * ramp
        elif end + fade <= len(audio):
            # Postroll at the outgoing endpoint has the same phase as the incoming first beat.
            mask = phase < fade
            pos = phase[mask]
            ramp = (pos / fade).astype(np.float32)[:, None]
            result[mask] = audio[end + pos] * (1 - ramp) + result[mask] * ramp
        else:
            # No context beyond either endpoint: microfades preserve the sample period.
            micro = min(round(0.003 * SR), fade)
            scale = np.minimum(1, np.minimum(phase / micro, (period - 1 - phase) / micro))
            result *= np.maximum(0, scale)[:, None]
    ramp_in = min(round(0.005 * SR), target // 4)
    ramp_out = min(round(0.015 * SR), target // 4)
    if ramp_in:
        result[:ramp_in] *= np.linspace(0, 1, ramp_in)[:, None]
    if ramp_out:
        result[-ramp_out:] *= np.cos(np.linspace(0, np.pi / 2, ramp_out))[:, None] ** 2
    return result
