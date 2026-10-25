"""Three effects buses; dynamic speech-band masking and exact placement before mastering."""
from __future__ import annotations

import json
import math
import shutil
import wave
from pathlib import Path

import numpy as np
from scipy.signal import istft, stft

from .dsp import _write_frames, read_dialogue_window, silence_envelope
from .media import PipelineError, SR, loudness, read_audio, write_audio
from .models import RunConfig, SpottingSheet
from .pacing_models import PacingPlan
from .rhythm import fit_music
from .sfx_engine import active_rms_db, crest_sample, level_effect, placement, sample_peak_db, window_loudness
from .event_scheduler import protected
from .sfx_engine import shape as shape_effect


def spectral_duck(effect: np.ndarray, dialogue: np.ndarray | None, tier: str) -> tuple[np.ndarray, dict]:
    if dialogue is None or len(effect) < 960:
        return effect, {"spectral_duck": "unavailable_without_dialogue_or_short_asset"}
    margins = {"ambience": (15, 4), "foley": (9, 5), "accent": (6, 6)}
    margin, maximum = margins[tier]
    args = dict(fs=SR, nperseg=960, noverlap=480, axis=0, boundary="zeros")
    frequencies, _, fx = stft(effect, **args)
    _, _, speech = stft(dialogue, **args)
    # Arrays: frequency, channel, time. Three broad adaptive speech bands.
    gains = np.ones((len(frequencies), 1, fx.shape[-1]))
    reductions = []
    for lo, hi in ((600, 1200), (1200, 2400), (2400, 5000)):
        band = (frequencies >= lo) & (frequencies < hi)
        d = 10 * np.log10(np.maximum(np.mean(np.abs(speech[band]) ** 2, axis=(0, 1)), 1e-12))
        x = 10 * np.log10(np.maximum(np.mean(np.abs(fx[band]) ** 2, axis=(0, 1)), 1e-12))
        # Activity threshold is a conservative STFT energy proxy, not speech recognition.
        control = np.where(d > -65, np.clip(d - margin - x, -maximum, 0), 0)
        current = 0.0
        for i, target in enumerate(control):
            tau = 0.02 if target < current else 0.15
            current = target + (current - target) * math.exp(-0.01 / tau)
            control[i] = current
        gains[band, 0, :] = 10 ** (control / 20)
        reductions.append(float(-control.min(initial=0)))
    _, output = istft(fx * gains, fs=SR, nperseg=960, noverlap=480, time_axis=-1, freq_axis=0)
    # scipy returns channels x samples with this axis selection.
    output = output.T[:len(effect)]
    if len(output) < len(effect):
        output = np.pad(output, ((0, len(effect) - len(output)), (0, 0)))
    return output.astype(np.float32), {"spectral_duck": "three_STFT_bands_600_to_5000Hz",
        "dialogue_margin_db": margin, "max_attenuation_db": maximum,
        "measured_max_band_reduction_db": reductions, "hop_sec": 0.01}


def render_effects(plan: PacingPlan, sheet: SpottingSheet, store, dialogue: Path | None,
                   root: Path, config: RunConfig) -> dict:
    """Disk-backed buses keep long-form memory bounded; source assets are never mutated."""
    count = round(sheet.video_duration_sec * SR)
    tiers = ("ambience", "foley", "accent")
    paths = {tier: root / f".{tier}.float" for tier in tiers}
    buses = {tier: np.memmap(paths[tier], mode="w+", dtype="float32", shape=(count, 2)) for tier in tiers}
    records = []
    try:
        for item in plan.effects:
            raw_path = root / "stems" / f"{item.asset_id}_sfx.wav"
            audio, rate = read_audio(raw_path)
            if rate != SR:
                raise PipelineError("Effects must be decoded at 48 kHz")
            audio = shape_effect(audio, item.duration_sec, item.category)
            ambience = item.tier == "ambience"
            if ambience:
                start = round(max(0, item.visual_time_sec - item.j_lead_sec) * SR)
                end = min(count, round(item.end_sec * SR))
                audio = fit_music(audio, (end - start) / SR, crossfade_ms=40)
                fade = min(round(item.crossfade_sec * SR), len(audio))
                if fade:
                    audio[:fade] *= np.sin(np.linspace(0, np.pi / 2, fade))[:, None]
                    audio[-fade:] *= np.cos(np.linspace(0, np.pi / 2, fade))[:, None]
            audio, level = level_effect(audio, item.target_lufs, item.peak_ceiling_db, ambience,
                level_mode=item.level_mode, target_peak_dbfs=item.target_peak_dbfs, target_rms_dbfs=item.target_rms_dbfs)
            base = audio.copy()
            pos = {"start_sample": start, "source_trim_samples": 0} if ambience else placement(
                item.visual_time_sec, item.av_offset_ms, crest_sample(audio))
            duck_info = {}
            # Re-evaluate landmark AFTER spectral shaping. Iterate because changing placement
            # slightly changes the speech sidechain. QC below catches non-convergence.
            for _ in range(4):
                raw_start = pos.get("unclipped_start_sample", pos["start_sample"])
                lead = max(0, -raw_start)
                voice = read_dialogue_window(dialogue, max(0, raw_start) / SR,
                                             max(0, len(base) - lead)) if dialogue else None
                if voice is not None and lead:
                    voice = np.pad(voice, ((lead, 0), (0, 0)))
                if (config.sfx_mix_profile == "audible" and item.tier == "accent" and
                        config.sfx_silence_policy == "accents" and protected(item.visual_time_sec, sheet)):
                    audio, duck_info = base.copy(), {"spectral_duck": "explicit_black_card_accent_unity"}
                else:
                    audio, duck_info = spectral_duck(base, voice, item.tier)
                if ambience:
                    break
                updated = placement(item.visual_time_sec, item.av_offset_ms, crest_sample(audio))
                if updated == pos:
                    break
                pos = updated
            audio = audio[pos["source_trim_samples"]:]
            start = pos["start_sample"]
            audio = audio[:max(0, count - start)]
            if not len(audio):
                raise PipelineError(f"Effect {item.event_id} falls outside the timeline")
            if not (item.tier == "accent" and config.sfx_silence_policy == "accents"):
                audio *= silence_envelope(len(audio), start / SR, sheet)[:, None]
            if np.max(np.abs(audio), initial=0) < 1e-8:
                records.append({"event_id": item.event_id, "status": "suppressed_by_silence", "passed": not item.required})
                continue
            prepared = root / "stems" / f"{item.asset_id}_prepared.wav"
            write_audio(prepared, audio)
            measured = loudness(prepared)
            tp = measured["input_tp"]
            if tp is None:
                raise PipelineError(f"Unmeasurable effect: {item.event_id}")
            if tp > item.peak_ceiling_db:
                audio *= 10 ** ((item.peak_ceiling_db - tp - 0.2) / 20)
                write_audio(prepared, audio)
                measured = loudness(prepared)
            audio, _ = read_audio(prepared)  # Quantized file is the actual placement source.
            peak = None if ambience else start + crest_sample(audio)
            target = pos.get("target_peak_sample")
            error_ms = None if ambience else (peak - target) / SR * 1000
            passed = (measured["input_tp"] <= item.peak_ceiling_db and
                      (ambience or abs(error_ms) <= 5))
            actual_peak, actual_rms = sample_peak_db(audio), active_rms_db(audio)
            prepared_lufs = window_loudness(audio, 3 if ambience else 0.4)
            level_error = item.target_lufs - prepared_lufs
            if item.level_mode == "sample_peak":
                level_error = item.target_peak_dbfs - actual_peak
            elif item.level_mode == "active_rms":
                level_error = item.target_rms_dbfs - actual_rms
            passed = passed and level_error <= 6.5
            buses[item.tier][start:start + len(audio)] += audio
            records.append({"event_id": item.event_id, "asset_id": item.asset_id,
                "tier": item.tier, "category": item.category, "status": "placed", "passed": passed,
                "start_sample": start, "source_trim_samples": pos["source_trim_samples"],
                "end_sample": start + len(audio), "target_peak_sample": target,
                "actual_peak_sample": peak, "landmark_error_ms": error_ms,
                "level": level, "spectral_control": duck_info, "file_true_peak_db": measured["input_tp"],
                "prepared_max_lufs": prepared_lufs,
                "sample_peak_dbfs": actual_peak, "active_rms_dbfs": actual_rms,
                "level_shortfall_db": max(0, level_error), "required": item.required,
                "prepared_file": str(prepared.relative_to(root)),
                "note": "RMS-crest sample landmark is a proxy for perceptual salience; inspect/listen."})
        bus_reports = {}
        for tier, bus in buses.items():
            # Summation can exceed an individual effect ceiling. Apply one constant bus
            # attenuation (no peak relocation), then verify the written bus with FFmpeg.
            ceiling = max((item.peak_ceiling_db for item in plan.effects if item.tier == tier),
                          default={"ambience": -10, "foley": -6, "accent": -7}[tier])
            maximum = max(float(np.max(np.abs(bus[a:a + SR * 10]), initial=0))
                          for a in range(0, count, SR * 10))
            gain = min(1, 10 ** (ceiling / 20) / max(maximum, 1e-12))
            path = root / f"{tier}.wav"
            _write_bus(path, bus, count, gain)
            measurement = loudness(path)
            if measurement["input_tp"] is not None and measurement["input_tp"] > ceiling:
                gain *= 10 ** ((ceiling - measurement["input_tp"] - 0.2) / 20)
                _write_bus(path, bus, count, gain)
                measurement = loudness(path)
            bus_reports[tier] = {"file": path.name, "attenuation_db": 20 * math.log10(gain),
                "true_peak_db": measurement["input_tp"], "ceiling_db": ceiling,
                "passed": measurement["input_tp"] is None or measurement["input_tp"] <= ceiling}
        combined = _combine(root, count)
        if combined["attenuation_db"] < 0:
            combined_gain = 10 ** (combined["attenuation_db"] / 20)
            for tier in tiers:
                _write_bus(root / f"{tier}.wav", buses[tier], count,
                           10 ** (bus_reports[tier]["attenuation_db"] / 20) * combined_gain)
                bus_reports[tier]["written_true_peak_db"] = loudness(root / f"{tier}.wav")["input_tp"]
            bridge, _ = read_audio(root / "bridge_sfx.wav")
            write_audio(root / "bridge_sfx.wav", bridge * combined_gain)
        shutil.copyfile(root / "sfx.wav", root / "sfx_stem.wav")
        for record in records:
            if record.get("status") == "placed":
                bus_gain = bus_reports[record["tier"]]["attenuation_db"] + combined["attenuation_db"]
                record["routed_sample_peak_dbfs"] = record["sample_peak_dbfs"] + bus_gain
                record["bus_attenuation_db"] = bus_gain
                if record["level"]["level_mode"] == "sample_peak":
                    shortfall = record["level"]["target_peak_dbfs"] - record["routed_sample_peak_dbfs"]
                    record["passed"] = record["passed"] and shortfall <= 6.5
                if record["level"]["level_mode"] == "active_rms":
                    shortfall = record["level"]["target_active_rms_dbfs"] - record["active_rms_dbfs"] - bus_gain
                    record["passed"] = record["passed"] and shortfall <= 6.5
                if record["level"]["level_mode"] == "loudness":
                    shortfall = record["level"]["target_lufs"] - record["prepared_max_lufs"] - bus_gain
                    record["passed"] = record["passed"] and shortfall <= 6.5
        report = {"effects": records, "buses": bus_reports, "silence_policy": config.sfx_silence_policy,
            "combined_bus": combined,
            "passed": combined["passed"] and all(x["passed"] for x in records) and all(x["passed"] for x in bus_reports.values()),
            "note": "Event/bus levels are pre-master. Final loudnorm changes all stems together."}
        (root / "sfx_qc.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        for bus in buses.values():
            bus.flush()
            bus._mmap.close()
        for path in paths.values():
            path.unlink(missing_ok=True)


def _write_bus(path, data, count, gain):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(SR)
        for a in range(0, count, SR * 10):
            _write_frames(out, data[a:a + SR * 10] * gain)


def _combine(root: Path, count: int):
    shutil.copyfile(root / "sfx.wav", root / "bridge_sfx.wav")
    sources = [wave.open(str(root / name), "rb") for name in
               ("bridge_sfx.wav", "ambience.wav", "foley.wav", "accent.wav")]
    output = root / "sfx_combined.wav"
    maximum = 0.0
    try:
        for a in range(0, count, SR * 10):
            block = min(count - a, SR * 10)
            data = sum(np.frombuffer(w.readframes(block), "<i2").astype(np.float32).reshape(-1, 2) / 32768
                       for w in sources)
            maximum = max(maximum, float(np.max(np.abs(data), initial=0)))
        gain = min(1, 10 ** (-1.7 / 20) / max(maximum, 1e-12))
        for w in sources:
            w.rewind()
        with wave.open(str(output), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(SR)
            for a in range(0, count, SR * 10):
                block = min(count - a, SR * 10)
                data = sum(np.frombuffer(w.readframes(block), "<i2").astype(np.float32).reshape(-1, 2) / 32768
                           for w in sources)
                _write_frames(out, data * gain)
    finally:
        for w in sources:
            w.close()
    output.replace(root / "sfx.wav")
    measured = loudness(root / "sfx.wav")
    tp = measured["input_tp"]
    if tp is not None and tp > -1.5:
        correction = 10 ** ((-1.7 - tp) / 20)
        data, _ = read_audio(root / "sfx.wav")
        write_audio(root / "sfx.wav", data * correction)
        gain *= correction
        tp = loudness(root / "sfx.wav")["input_tp"]
    return {"nonzero": maximum > 1e-6, "attenuation_db": 20 * math.log10(gain),
            "true_peak_db": tp, "ceiling_db": -1.5,
            "passed": maximum > 1e-6 and tp is not None and tp <= -1.5}
