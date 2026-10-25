"""Optional narration chain; original archival audio remains sample-identical inside its windows."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .media import PipelineError, SR, ffmpeg, loudness, normalize, read_audio, write_audio

CHAIN = (
    "[0:a]highpass=f=75:p=2,equalizer=f=280:t=q:w=1.2:g=-2.5,"
    "equalizer=f=3200:t=q:w=1:g=1.5,"
    "acompressor=threshold=0.08:ratio=2:attack=15:release=120:makeup=1,"
    "acrossover=split=6500:order=4th[low][high];"
    "[high]acompressor=threshold=0.025:ratio=4:attack=3:release=60:makeup=1[deessed];"
    "[low][deessed]amix=inputs=2:normalize=0[out]"
)


def polish_dialogue(source: Path, windows, root: Path):
    original, rate = read_audio(source)
    if rate != SR:
        raise PipelineError("Dialogue polish requires 48kHz source")
    keep = np.ones(len(original), np.float32)
    for window in windows:
        a, b = round(window.start_sec * SR), round(window.end_sec * SR)
        keep[a:b] = 0
    if not np.any(np.abs(original[keep > 0]) > 1e-5):
        return source, {"status": "skipped_no_narration_outside_archives"}
    analysis = root / "narration_only.wav"
    processed = root / "narration_chain.wav"
    normalized = root / "narration_normalized.wav"
    write_audio(analysis, original * keep[:, None])
    ffmpeg("-i", analysis, "-filter_complex", CHAIN, "-map", "[out]", "-ar", SR, "-c:a", "pcm_f32le", processed)
    normalize(processed, normalized, -14, -2)
    normalized_qc = loudness(normalized)
    decoded = root / "narration_normalized_pcm16.wav"
    ffmpeg("-i", normalized, "-ar", SR, "-ac", 2, "-c:a", "pcm_s16le", decoded)
    polished, _ = read_audio(decoded)
    polished = np.pad(polished[:len(original)], ((0, max(0, len(original) - len(polished))), (0, 0)))
    blend = keep.copy()
    fade = round(0.015 * SR)
    for window in windows:
        a, b = round(window.start_sec * SR), round(window.end_sec * SR)
        left, right = max(0, a - fade), min(len(original), b + fade)
        blend[left:a] = np.minimum(blend[left:a], np.linspace(1, 0, a - left))
        blend[b:right] = np.minimum(blend[b:right], np.linspace(0, 1, right - b))
    result = polished * blend[:, None] + original * (1 - blend[:, None])
    output = root / "dialogue_polished.wav"
    write_audio(output, result)
    actual, _ = read_audio(output)
    preserved = all(np.array_equal(actual[round(w.start_sec * SR):round(w.end_sec * SR)],
                                   original[round(w.start_sec * SR):round(w.end_sec * SR)]) for w in windows)
    if not preserved:
        raise PipelineError("Dialogue polish altered protected archival samples")
    return output, {"status": "processed", "archival_samples_preserved": preserved,
        "passed": (normalized_qc["input_i"] is not None and abs(normalized_qc["input_i"] + 14) <= 1 and
                   normalized_qc["input_tp"] is not None and normalized_qc["input_tp"] <= -1),
        "narration_only_measurement": normalized_qc, "target_narration_lufs": -14,
        "chain": "75Hz HPF; 280Hz -2.5dB/Q1.2; 3.2kHz +1.5dB/Q1; 2:1/15ms/120ms; 6.5kHz split de-ess",
        "note": "Integrated narration target, not fixed per-word loudness; EQ/compression do not remove room echo"}
