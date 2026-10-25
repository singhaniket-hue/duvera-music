"""Prove that the actual FFmpeg premix includes the SFX waveform at unity gain."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .media import PipelineError, SR


def verify_sfx_routing(premix: Path, reference: Path, sfx: Path) -> dict:
    rate, mixed = wavfile.read(premix, mmap=True)
    reference_rate, without = wavfile.read(reference, mmap=True)
    if rate != SR or reference_rate != SR or mixed.shape != without.shape:
        raise PipelineError("Premix routing QC clock/shape mismatch")
    maximum_error, expected_power, actual_power, samples = 0., 0., 0., 0
    with wave.open(str(sfx), "rb") as source:
        if source.getframerate() != SR or source.getnframes() != len(mixed):
            raise PipelineError("SFX bus is not timeline-length 48kHz PCM")
        for start in range(0, len(mixed), SR * 10):
            raw = source.readframes(min(SR * 10, len(mixed) - start))
            expected = np.frombuffer(raw, "<i2").reshape(-1, 2).astype(np.float64) / 32768
            actual = mixed[start:start + len(expected)].astype(np.float64) - without[start:start + len(expected)]
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected), initial=0)))
            expected_power += float(np.sum(expected ** 2))
            actual_power += float(np.sum(actual ** 2))
            samples += expected.size
    return {"passed": maximum_error < 2e-6, "max_sample_error": maximum_error,
        "expected_sfx_rms_dbfs": float(10 * np.log10(max(expected_power / max(1, samples), 1e-12))),
        "actual_sfx_rms_dbfs": float(10 * np.log10(max(actual_power / max(1, samples), 1e-12))),
        "nonzero": expected_power > 1e-10,
        "contract": "premix minus actual dialogue+music reference equals SFX stem at unity; before mastering"}
