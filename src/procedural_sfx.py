"""Explicit, deterministic documentary accents; never a substitute for music or object Foley."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import butter, sosfilt

from .media import SR, write_audio

CATEGORIES = {"sub_drop", "paper_slide", "ui_tick", "text_pop", "whoosh", "riser", "shutter"}


def synthesize(category, duration_sec, destination, seed_key=""):
    if category not in CATEGORIES:
        raise ValueError("No procedural generator for this sound family")
    length = max(0.035, min(2., duration_sec))
    t = np.arange(round(length * SR)) / SR
    rng = np.random.default_rng(int(hashlib.sha256((category + seed_key).encode()).hexdigest()[:16], 16))
    if category == "sub_drop":
        frequency = 36 + 49 * np.exp(-6.5 * t)
        phase = np.cumsum(frequency) * (2 * np.pi / SR)
        signal = np.sin(phase) * np.minimum(t / 0.012, 1) * np.exp(-4.2 * t)
    elif category in {"paper_slide", "text_pop"}:
        phase = 2 * np.pi * (110 * t + 130 / 40 * (1 - np.exp(-40 * t)))
        noise = sosfilt(butter(2, [400, 3500], "bandpass", fs=SR, output="sos"), rng.normal(0, 0.2, len(t)))
        signal = 0.8 * np.sin(phase) * np.exp(-35 * t) + 0.25 * noise * np.exp(-((t - 0.018) / 0.018) ** 2)
    elif category in {"whoosh", "riser"}:
        noise = sosfilt(butter(2, [180, 4500], "bandpass", fs=SR, output="sos"), rng.normal(0, 0.3, len(t)))
        crest = length * (0.8 if category == "riser" else 0.65)
        signal = noise * np.exp(-((t - crest) / max(0.02, length * 0.2)) ** 2)
    elif category == "shutter":
        noise = sosfilt(butter(2, [500, 4200], "bandpass", fs=SR, output="sos"), rng.normal(0, 0.25, len(t)))
        signal = noise * (np.exp(-((t - 0.015) / 0.006) ** 2) + 0.5 * np.exp(-((t - 0.045) / 0.01) ** 2))
    else:
        signal = np.sin(2 * np.pi * 2400 * t) * np.exp(-140 * t)
    signal *= np.minimum(t / 0.002, 1)
    signal *= np.minimum(np.maximum(0, length - t) / 0.012, 1)
    write_audio(destination, np.column_stack([signal, signal]).astype(np.float32) * 0.5)
    return destination


def procedural_jobs(pacing, config):
    if not config.procedural_sfx or pacing is None:
        return {}
    return {e.asset_id: e for e in pacing.effects if e.tier == "accent" and e.category in CATEGORIES}
