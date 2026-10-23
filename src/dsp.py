"""Bounded momentary-loudness control, spectral separation, click-free edits and QC."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from .analyzer import onset_analysis
from .media import SR, ffmpeg, loudness, normalize, read_audio, write_audio
from .models import MusicCue, RunConfig, SpottingSheet


def k_weighted(audio: np.ndarray) -> np.ndarray:
    """BS.1770 K-weighting coefficients for exactly 48 kHz, stereo without surround weights."""
    stage1 = lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285],
                     [1, -1.69065929318241, 0.73248077421585], audio, axis=0)
    return lfilter([1, -2, 1], [1, -1.99004745483398, 0.99007225036621], stage1, axis=0)


def momentary(audio: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """400 ms energy windows at 100 ms hops; no integrated-loudness gating."""
    weighted = k_weighted(audio)
    power = np.sum(weighted.astype(np.float64) ** 2, axis=1)
    window, hop = round(0.4 * sr), round(0.1 * sr)
    if not len(power):
        return np.array([0.0]), np.array([-120.0])
    starts = np.arange(0, max(1, len(power) - window + 1), hop)
    ends = np.minimum(starts + window, len(power))
    summed = np.concatenate(([0.0], np.cumsum(power)))
    energy = (summed[ends] - summed[starts]) / np.maximum(1, ends - starts)
    return (starts + (ends - starts) / 2) / sr, -0.691 + 10 * np.log10(np.maximum(energy, 1e-12))


def interpolate(times: np.ndarray, values: np.ndarray, count: int, sr: int = SR) -> np.ndarray:
    return np.interp(np.arange(count) / sr, times, values).astype(np.float32)


def fit_music(audio: np.ndarray, seconds: float, onset: float = 0) -> np.ndarray:
    """Loop short source with crossfades; exact timeline length; short sample-safe fades."""
    target = round(seconds * SR)
    start = min(max(0, round(onset * SR)), max(0, len(audio) - 1))
    source = audio[start:]
    if len(source) == 0:
        raise ValueError("Empty music asset")
    result = np.zeros((target, 2), dtype=np.float32)
    length = min(target, len(source))
    result[:length] = source[:length]
    cursor = length
    overlap = min(round(0.1 * SR), len(source) // 4)
    while cursor < target:
        cross = min(overlap, cursor)
        take = min(len(source), target - cursor + cross)
        if cross:
            ramp = np.linspace(0, 1, cross, dtype=np.float32)[:, None]
            result[cursor - cross:cursor] = result[cursor - cross:cursor] * (1 - ramp) + source[:cross] * ramp
        result[cursor:cursor + take - cross] = source[cross:take]
        cursor += take - cross
    ramp_in = min(round(0.005 * SR), target // 4)
    ramp_out = min(round(0.015 * SR), target // 4)
    if ramp_in:
        result[:ramp_in] *= np.linspace(0, 1, ramp_in)[:, None]
    if ramp_out:
        result[-ramp_out:] *= np.cos(np.linspace(0, math.pi / 2, ramp_out))[:, None] ** 2
    return result


def silence_envelope(count: int, offset: float, sheet: SpottingSheet) -> np.ndarray:
    t = offset + np.arange(count) / SR
    gain = np.ones(count, dtype=np.float32)
    for silence in sheet.silence_windows:
        # Fade OUT before the protected line, fade IN after it; silence itself is exactly zero.
        a, b, fade = silence.start_sec, silence.end_sec, 0.03
        local = np.ones(count, dtype=np.float32)
        down = (t >= a - fade) & (t < a)
        up = (t > b) & (t <= b + fade)
        local[down] = (a - t[down]) / fade
        local[(t >= a) & (t <= b)] = 0
        local[up] = (t[up] - b) / fade
        gain = np.minimum(gain, local)
    return gain


def duck(music: np.ndarray, dialogue: np.ndarray | None, cue: MusicCue) -> tuple[np.ndarray, dict]:
    cutoffs = {"barely_behind": 12000, "background": 9500,
               "cinematic_distant": 6500, "montage_foreground": 16000}
    filtered = sosfilt(butter(2, 45, "highpass", fs=SR, output="sos"), music, axis=0)
    filtered = sosfilt(butter(2, cutoffs[cue.depth_profile], "lowpass", fs=SR, output="sos"), filtered, axis=0)
    times, music_l = momentary(filtered)
    valid_music = music_l > -70
    baseline = float(np.median(music_l[valid_music])) if valid_music.any() else -60
    base_gain = float(np.clip(-25 - baseline, -36, 6))
    if dialogue is None:
        return (filtered * 10 ** (base_gain / 20)).astype(np.float32), {
            "base_gain_db": base_gain, "gap_control": "unavailable_without_dialogue"}
    _, speech_l = momentary(dialogue)
    active = speech_l > -55
    activity = interpolate(times, active.astype(float), len(music))
    # Preserve stereo side information, dynamically carve center presence by up to 3.5 dB.
    mid, side = (filtered[:, 0] + filtered[:, 1]) / 2, (filtered[:, 0] - filtered[:, 1]) / 2
    band = sosfilt(butter(2, [1500, 4000], "bandpass", fs=SR, output="sos"), mid)
    mid -= band * activity * (1 - 10 ** (-3.5 / 20))
    filtered = np.column_stack([mid + side, mid - side]).astype(np.float32)
    _, music_l = momentary(filtered)
    desired = np.where(active, np.minimum(base_gain, speech_l - cue.target_lu_gap - music_l), base_gain)
    desired = np.clip(desired, base_gain - 12, base_gain)
    # Frame-domain attack/release; immediate onset protection is helped by centered 400ms lookahead.
    smoothed = np.empty_like(desired)
    current = float(desired[0])
    for i, value in enumerate(desired):
        tau = 0.02 if value < current else 0.25
        current = value + (current - value) * math.exp(-0.1 / tau)
        smoothed[i] = current
    filtered *= (10 ** (interpolate(times, smoothed, len(music)) / 20))[:, None]
    _, actual_l = momentary(filtered)
    valid = active & (actual_l > -70)
    gaps = speech_l[valid] - actual_l[valid]
    return filtered, {
        "base_gain_db": base_gain, "max_dynamic_reduction_db": 12,
        "target_gap_lu": cue.target_lu_gap,
        "measured_gap_p10_lu": round(float(np.percentile(gaps, 10)), 2) if len(gaps) else None,
        "windows_below_target": int(np.sum(gaps < cue.target_lu_gap - 1)),
        "measured_windows": len(gaps),
        "gap_control": "400ms K-weighted windows; bounded attenuation, review deviations",
    }


def read_dialogue_window(path: Path, start: float, count: int) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        position = min(round(start * SR), w.getnframes())
        w.setpos(position)
        raw = np.frombuffer(w.readframes(count), "<i2").astype(np.float32).reshape(-1, 2) / 32768
    if len(raw) < count:
        raw = np.pad(raw, ((0, count - len(raw)), (0, 0)))
    return raw


def _write_frames(writer, audio):
    writer.writeframes((np.clip(audio, -1, 1 - 1 / 32768) * 32768).astype("<i2").tobytes())


def build_stems(sheet: SpottingSheet, store, dialogue: Path | None, root: Path) -> tuple[Path, Path, list]:
    bed_path, sfx_path = root / "music.wav", root / "sfx.wav"
    records, cursor = [], 0
    with wave.open(str(bed_path), "wb") as bed, wave.open(str(sfx_path), "wb") as sfx:
        for w in (bed, sfx):
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
        for cue in sheet.cues:
            start, end = round(cue.start_sec * SR), round(cue.end_sec * SR)
            gap = start - cursor
            while gap > 0:
                block = min(gap, SR * 10)
                for w in (bed, sfx):
                    _write_frames(w, np.zeros((block, 2), np.float32))
                gap -= block
            source = root / "stems" / f"{cue.cue_id}_music.wav"
            analysis = onset_analysis(source)
            raw, _ = read_audio(source)
            manual_downbeat = store.assets["downbeats"].get(cue.cue_id)
            if manual_downbeat is not None:
                if not 0 <= float(manual_downbeat) < len(raw) / SR:
                    raise ValueError("Supplied downbeat offset outside asset")
                entry = float(manual_downbeat)
                analysis["downbeat_verified"] = "user_supplied"
            else:
                # Skip only a weak, short intro; onsets are NOT bar downbeats.
                first = analysis["onsets"][0] if analysis["onsets"] else 0
                entry = first if first < 3 else 0
            seconds = (end - start) / SR
            music = fit_music(raw, seconds, entry)
            if cue.exit_type == "transient_stop":
                onsets = [x - entry for x in analysis["onsets"] if seconds - 0.65 <= x - entry <= seconds - 0.09]
                if onsets:
                    stop = min(len(music), round((onsets[-1] + 0.09) * SR))
                    fade = min(round(0.015 * SR), stop)
                    if fade:
                        music[stop - fade:stop] *= np.linspace(1, 0, fade)[:, None]
                    music[stop:] = 0
            voice = read_dialogue_window(dialogue, cue.start_sec, end - start) if dialogue else None
            processed, metrics = duck(music, voice, cue)
            silence = silence_envelope(len(processed), cue.start_sec, sheet)
            processed *= silence[:, None]
            # All exits fade, including filter ringing introduced after the initial clip fade.
            ramp = min(round(0.015 * SR), len(processed) // 4)
            if ramp:
                processed[-ramp:] *= np.linspace(1, 0, ramp)[:, None]
            peak = float(np.max(np.abs(processed), initial=0))
            headroom_gain = min(1.0, 0.98 / max(peak, 1e-12))
            processed *= headroom_gain
            metrics["headroom_attenuation_db"] = round(20 * math.log10(headroom_gain), 3)
            fx = np.zeros_like(processed)
            if cue.bridge_sfx_prompt:
                raw_fx, _ = read_audio(root / "stems" / f"{cue.cue_id}_sfx.wav")
                take = min(len(fx), len(raw_fx), 2 * SR)
                effect = fit_music(raw_fx, take / SR)
                fx[-take:] = effect * 10 ** (-24 / 20)
                fx *= silence[:, None]
            prepared = root / "stems" / f"{cue.cue_id}_prepared.wav"
            write_audio(prepared, processed)
            _write_frames(bed, processed)
            _write_frames(sfx, fx)
            cursor = end
            overlap_db = None
            if voice is not None:
                band_sos = butter(2, [1000, 4000], "bandpass", fs=SR, output="sos")
                v = np.mean(sosfilt(band_sos, voice, axis=0) ** 2)
                m = np.mean(sosfilt(band_sos, processed, axis=0) ** 2)
                if v > 1e-12 and m > 1e-12:
                    overlap_db = round(float(10 * np.log10(m / v)), 2)
            records.append({"cue_id": cue.cue_id, "entry_offset_sec": entry, "analysis": analysis,
                            "control": metrics, "speech_band_music_to_voice_db": overlap_db})
        tail = round(sheet.video_duration_sec * SR) - cursor
        while tail > 0:
            count = min(tail, SR * 10)
            for w in (bed, sfx):
                _write_frames(w, np.zeros((count, 2), np.float32))
            tail -= count
    return bed_path, sfx_path, records


def render(dialogue: Path, music: Path, sfx: Path, root: Path, config: RunConfig) -> dict:
    premix, master = root / "premix.wav", root / "master.wav"
    ffmpeg("-i", dialogue, "-i", music, "-i", sfx,
           "-filter_complex", "[0:a]highpass=f=65:p=2[v];[v][1:a][2:a]amix=inputs=3:duration=first:normalize=0[m]",
           "-map", "[m]", "-ar", SR, "-c:a", "pcm_f32le", premix)
    normalize(premix, master, config.target_lufs, config.true_peak_db)
    master_qc = loudness(master)
    video = root / f"final_video_{root.name}.mp4"
    # Re-encode for predictable MP4 compatibility, preserving image dimensions and frame rate.
    ffmpeg("-i", config.video_path, "-i", master, "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:a", "aac", "-b:a", "192k",
           "-ar", SR, "-movflags", "+faststart", "-shortest", video)
    # Measure the actual encoded output: AAC can change true peaks.
    final_qc = loudness(video)
    actual_i, actual_tp = final_qc.get("input_i"), final_qc.get("input_tp")
    passed = (actual_i is not None and actual_tp is not None and
              abs(actual_i - config.target_lufs) <= 1 and actual_tp <= -1)
    premix.unlink(missing_ok=True)
    return {"master": master.name, "video": video.name, "master_measurement": master_qc,
            "encoded_measurement": final_qc, "passed": passed,
            "target_lufs": config.target_lufs, "target_peak_db": config.true_peak_db,
            "qc_peak_ceiling_db": -1,
            "note": "QC failure requires review; a file is not upload-ready merely because it rendered."}
