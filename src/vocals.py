"""Fail-closed vocal-stem proxy gate. Separation energy is not a singing classifier."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .media import PipelineError, SR, ffmpeg, read_audio, run, write_audio
from .models import RunConfig


def db_rms(audio: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 1e-12)))


def read_float(path: Path) -> tuple[np.ndarray, int]:
    rate, data = wavfile.read(path)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / (np.iinfo(data.dtype).max + 1)
    if data.ndim == 1:
        data = data[:, None]
    if not len(data) or not np.isfinite(data).all():
        raise PipelineError("Empty or non-finite separator output")
    return data.astype(np.float32), rate


def vocal_scores(mix: np.ndarray, vocals: np.ndarray, rate: int) -> dict:
    if len(mix) != len(vocals):
        raise PipelineError("Separator output is not aligned to the input")
    if not np.isfinite(mix).all() or not np.isfinite(vocals).all():
        raise PipelineError("Non-finite vocal-gate input")
    mix_level, vocal_level = db_rms(mix), db_rms(vocals)
    windows = []
    width, hop = max(1, round(0.4 * rate)), max(1, round(0.2 * rate))
    for start in range(0, len(mix), hop):
        a, b = mix[start:start + width], vocals[start:start + width]
        if len(a) < min(width // 2, len(mix)):
            continue
        local_mix = db_rms(a)
        if local_mix > max(-90, mix_level - 40):
            # Scale-invariant proxy: each active window's mixture is referenced to -18 dBFS RMS.
            windows.append(db_rms(b) - local_mix - 18)
    return {
        "mix_rms_dbfs": mix_level, "vocal_stem_rms_dbfs": vocal_level,
        "vocal_to_mix_db": vocal_level - mix_level,
        "max_window_vocal_at_minus18_mix_dbfs": max(windows, default=-240),
        "active_windows": len(windows),
        "separator_output_clipped": bool(np.max(np.abs(vocals), initial=0) >= 0.989),
        "note": "Source-separation energy proxy, not proof of singing or vocal absence.",
    }


class DemucsSeparator:
    def __init__(self, config: RunConfig):
        self.config = config

    def prepare(self):
        if any(importlib.util.find_spec(x) is None for x in ("demucs", "torch", "torchaudio")):
            raise PipelineError("Vocal checking requires the vocals extra: pip install -e '.[vocals]'. "
                                "Default policy blocks mixing when the checker is unavailable.")
        # Load/cache the selected model and verify the device BEFORE credit-consuming generation.
        run([sys.executable, "-c",
             "import sys; from demucs.pretrained import get_model; get_model(sys.argv[1]).to(sys.argv[2])",
             self.config.vocal_model, self.config.vocal_device], timeout=900)

    def separate(self, source: Path, output: Path) -> tuple[Path, Path]:
        run([sys.executable, "-m", "demucs.separate", "--two-stems", "vocals",
             "--float32", "--clip-mode", "clamp", "-n", self.config.vocal_model,
             "-d", self.config.vocal_device, "-o", output, source], timeout=1800)
        folder = output / self.config.vocal_model / source.stem
        vocals, backing = folder / "vocals.wav", folder / "no_vocals.wav"
        if not vocals.is_file() or not backing.is_file():
            raise PipelineError("Demucs did not produce vocals.wav and no_vocals.wav")
        return vocals, backing


class VocalGuard:
    def __init__(self, config: RunConfig, root: Path, separator=None):
        self.config, self.root = config, root
        self.separator = separator or DemucsSeparator(config)
        self.reports, self.cache = {}, {}
        root.mkdir(parents=True, exist_ok=True)
        self.save()

    def prepare(self):
        if self.config.vocal_policy != "off":
            self.separator.prepare()

    def save(self):
        payload = {"policy": self.config.vocal_policy, "model": self.config.vocal_model,
                   "threshold_dbfs_at_minus18_mix": self.config.vocal_threshold_dbfs,
                   "assets": self.reports,
                   "limitations": "Demucs can leak instruments or miss quiet vocals. Human review remains necessary."}
        (self.root / "vocal_qc.json").write_text(json.dumps(payload, indent=2, allow_nan=False))

    def _measure(self, source: Path, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        original, _ = read_audio(source)
        peak = float(np.max(abs(original), initial=0))
        if peak < 1e-8:
            raise PipelineError("Music asset is silent; replace it rather than treating it as a vocal pass")
        gain = min(-18 - db_rms(original), -6 - 20 * np.log10(peak))
        normalized = folder / "analysis.wav"
        ffmpeg("-i", source, "-af", f"volume={gain}dB", "-ar", SR, "-ac", "2",
               "-c:a", "pcm_f32le", normalized)
        vocal_path, backing = self.separator.separate(normalized, folder / "separated")
        voice, rate = read_float(vocal_path)
        mix, mix_rate = read_float(normalized)
        if rate != mix_rate:
            divisor = math.gcd(rate, mix_rate)
            mix = resample_poly(mix, rate // divisor, mix_rate // divisor)
        if abs(len(mix) - len(voice)) > max(2, round(0.02 * rate)):
            raise PipelineError("Separator changed audio duration")
        size = min(len(mix), len(voice))
        return vocal_scores(mix[:size], voice[:size], rate), backing, float(gain), len(original)

    def _passes(self, scores: dict) -> bool:
        return (not scores["separator_output_clipped"] and
                scores["max_window_vocal_at_minus18_mix_dbfs"] <= self.config.vocal_threshold_dbfs)

    def inspect(self, cue_id: str, source: Path) -> Path:
        if self.config.vocal_policy == "off":
            self.reports[cue_id] = {"status": "disabled_unchecked", "source": source.name}
            self.save()
            return source
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in self.cache:
            path, report = self.cache[digest]
            self.reports[cue_id] = {**report, "reused_content_check": True}
            self.save()
            return path
        report = {"status": "checking", "source_sha256": digest, "source": source.name}
        self.reports[cue_id] = report
        self.save()
        try:
            scores, backing, gain, count = self._measure(source, self.root / digest[:20] / "original")
            report["original"] = scores
            selected = source
            if not self._passes(scores):
                if self.config.vocal_policy == "reject":
                    report["status"] = "rejected"
                    raise PipelineError(f"{cue_id}: vocal-stem proxy exceeds the limit or clipped. "
                                        "Replace it, or import the retained raw stem into a new --vocal-policy strip run. "
                                        "No automatic paid regeneration was attempted.")
                # Strip ONLY the music asset. Narration is never passed to the separator.
                selected = self.root / digest[:20] / "instrumental.wav"
                backing_audio, _ = read_float(backing)
                if np.max(abs(backing_audio), initial=0) >= 0.989:
                    raise PipelineError("Separated backing may be clipped; replace the music")
                ffmpeg("-i", backing, "-af", f"volume={-gain}dB", "-ac", "2", "-ar", SR,
                       "-c:a", "pcm_f32le", selected.with_suffix(".float.wav"))
                cleaned, _ = read_float(selected.with_suffix(".float.wav"))
                if abs(len(cleaned) - count) > round(0.02 * SR):
                    raise PipelineError("Stripped music changed duration")
                cleaned = np.pad(cleaned[:count], ((0, max(0, count - len(cleaned))), (0, 0)))
                cleaned *= min(1, 0.98 / max(float(np.max(abs(cleaned), initial=0)), 1e-12))
                write_audio(selected, cleaned)
                residual, _, _, _ = self._measure(selected, self.root / digest[:20] / "residual")
                report["residual"] = residual
                if not self._passes(residual):
                    report["status"] = "rejected_residual"
                    raise PipelineError(f"{cue_id}: stripped music still exceeds the vocal proxy limit; replace it.")
                report["status"] = "stripped_and_rechecked"
            else:
                report["status"] = "below_proxy_threshold"
            report["selected_file_relative_to_run"] = os.path.relpath(selected, self.root.parent)
            report["selected_sha256"] = hashlib.sha256(selected.read_bytes()).hexdigest()
            self.cache[digest] = (selected, dict(report))
            return selected
        except Exception:
            if report["status"] == "checking":
                report["status"] = "checker_error"
            raise
        finally:
            self.save()
