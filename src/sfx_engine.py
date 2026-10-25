"""Shape a COPY of an effect, meter it, then align its acoustic landmark on the sample clock."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .dsp import k_weighted
from .media import PipelineError, SR


def crest_sample(audio: np.ndarray) -> int:
    if not len(audio) or not np.isfinite(audio).all():
        raise PipelineError("Empty or invalid SFX asset")
    power = np.mean(audio.astype(np.float64) ** 2, axis=1)
    if power.max(initial=0) < 1e-12:
        raise PipelineError("Silent SFX asset has no acoustic landmark")
    # Centred 2ms RMS crest, refined to a sample-energy peak within that neighbourhood.
    width = round(0.002 * SR)
    envelope = np.convolve(power, np.ones(width) / width, mode="same")[:len(power)]
    centre = int(np.argmax(envelope))
    a, b = max(0, centre - width // 2), min(len(power), centre + width // 2 + 1)
    return a + int(np.argmax(power[a:b]))


def shape(audio: np.ndarray, duration_sec: float, category: str) -> np.ndarray:
    audio = np.asarray(audio, np.float32).copy()
    if category == "sub_drop":
        audio = sosfilt(butter(4, [35, 90], "bandpass", fs=SR, output="sos"), audio, axis=0)
    else:
        cutoff = 60 if category == "ambience" else 90
        audio = sosfilt(butter(2, cutoff, "highpass", fs=SR, output="sos"), audio, axis=0)
    count = max(1, round(duration_sec * SR))
    if len(audio) > count and category != "ambience":
        peak = crest_sample(audio)
        pre = min(peak, round(0.025 * SR), count // 4)
        start = max(0, min(len(audio) - count, peak - pre))
        audio = audio[start:start + count].copy()
    elif category != "ambience":
        audio = audio[:count].copy()
    fade_in = min(round(0.003 * SR), len(audio) // 4)
    fade_out = min(round(0.02 * SR), len(audio) // 3)
    if fade_in:
        audio[:fade_in] *= np.sin(np.linspace(0, np.pi / 2, fade_in))[:, None] ** 2
    if fade_out:
        audio[-fade_out:] *= np.cos(np.linspace(0, np.pi / 2, fade_out))[:, None] ** 2
    crest_sample(audio)  # Fail on silent, failed synthesis instead of reporting a fictional peak.
    return audio.astype(np.float32)


def window_loudness(audio: np.ndarray, seconds: float) -> float:
    """Maximum K-weighted fixed-window LUFS; short assets zero-pad to the FULL window."""
    weighted = k_weighted(audio)
    power = np.sum(weighted.astype(np.float64) ** 2, axis=1)
    width, hop = round(seconds * SR), round(0.01 * SR)
    power = np.pad(power, (0, max(0, width - len(power))))
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, len(power) - width + 1, hop)
    starts = np.unique(np.append(starts, len(power) - width))
    maximum = np.max((cumulative[starts + width] - cumulative[starts]) / width, initial=0)
    return float(-0.691 + 10 * np.log10(max(maximum, 1e-12)))


def peak_db(audio: np.ndarray) -> float:
    """4x polyphase reconstruction estimate; written files are also checked with FFmpeg."""
    peak = np.max(np.abs(resample_poly(audio, 4, 1, axis=0)), initial=0)
    return float(20 * np.log10(max(float(peak), 1e-12)))


def sample_peak_db(audio: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.max(np.abs(audio), initial=0)), 1e-12)))


def active_rms_db(audio: np.ndarray) -> float:
    """Linked stereo RMS in active 20ms blocks, excluding tails >30dB below the strongest block."""
    width = round(0.02 * SR)
    padded = np.pad(audio, ((0, (-len(audio)) % width), (0, 0)))
    power = np.mean(padded.astype(np.float64).reshape(-1, width, 2) ** 2, axis=(1, 2))
    active = power > max(1e-8, float(power.max(initial=0)) * 1e-3)
    return float(10 * np.log10(max(float(power[active].mean()) if active.any() else 0, 1e-12)))


def level_effect(audio: np.ndarray, target_lufs: float, ceiling: float, ambience=False, *,
                 level_mode="loudness", target_peak_dbfs=None, target_rms_dbfs=None) -> tuple[np.ndarray, dict]:
    window = 3.0 if ambience else 0.4
    measured = window_loudness(audio, window)
    requested_gain = target_lufs - measured
    if level_mode == "sample_peak":
        requested_gain = target_peak_dbfs - sample_peak_db(audio)
    elif level_mode == "active_rms":
        requested_gain = target_rms_dbfs - active_rms_db(audio)
    gain = min(requested_gain, ceiling - 0.2 - peak_db(audio), 36)
    result = (audio * 10 ** (gain / 20)).astype(np.float32)
    return result, {"meter": "max_short_term_3s" if ambience else "max_momentary_400ms",
                    "window_sec": window, "short_assets_zero_padded": True,
                    "target_lufs": target_lufs, "measured_lufs": window_loudness(result, window),
                    "level_mode": level_mode, "target_peak_dbfs": target_peak_dbfs,
                    "target_active_rms_dbfs": target_rms_dbfs,
                    "sample_peak_dbfs": sample_peak_db(result), "active_rms_dbfs": active_rms_db(result),
                    "reconstructed_peak_estimate_db": peak_db(result), "gain_db": gain}


def placement(visual_time: float, av_offset_ms: float, peak_sample: int) -> dict:
    target = round(visual_time * SR) + round(av_offset_ms * SR / 1000)
    raw_start = target - peak_sample
    return {"target_peak_sample": target, "start_sample": max(0, raw_start),
            "source_trim_samples": max(0, -raw_start), "unclipped_start_sample": raw_start}
