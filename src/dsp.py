"""Bounded momentary-loudness control, spectral separation, click-free edits and QC."""
from __future__ import annotations

import math
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from .analyzer import onset_analysis
from .media import PipelineError, SR, ffmpeg, loudness, normalize, probe, read_audio, write_audio
from .models import MusicCue, RunConfig, SpottingSheet
from .narrative import fully_silent
from .rhythm import fit_music, plan_loop


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


def transient_exit(audio: np.ndarray, tail_ms: float, anchor_before_sec: float | None = None) -> tuple[np.ndarray, dict]:
    """Choose a strong late onset and fit decay/reverb BEFORE the protected boundary."""
    count = len(audio)
    tail = round(tail_ms / 1000 * SR)
    decay = round(0.09 * SR)
    hop = round(0.01 * SR)
    first = max(0, count - round(0.8 * SR))
    last = count - tail - decay
    if last <= first or last - first < hop * 2:
        return audio, {"method": "microfade_only", "tail_ms": 0}
    energy = np.mean(audio[first:last] ** 2, axis=1)
    blocks = len(energy) // hop
    rms = np.sqrt(energy[:blocks * hop].reshape(blocks, hop).mean(axis=1))
    flux = np.maximum(0, np.diff(rms, prepend=rms[0]))
    if flux.max(initial=0) < 1e-6:
        return audio, {"method": "microfade_only", "tail_ms": 0}
    target = (count / SR - anchor_before_sec) if anchor_before_sec is not None else None
    salience = flux
    if target is not None:
        times = (first + np.arange(len(flux)) * hop) / SR
        # Prefer the actual pluck nearest the requested pre-boundary accent, never synthesize one.
        salience = flux * np.exp(-0.5 * ((times - target) / 0.12) ** 2)
    onset = first + int(np.argmax(salience)) * hop
    dry_end = min(count, onset + decay)
    result = audio.copy()
    source = audio[max(onset, dry_end - round(0.05 * SR)):dry_end]
    result[dry_end:] = 0
    # Short tapped decay from the actual transient, bounded by the cue endpoint.
    for delay_ms, level in ((35, 0.18), (70, 0.12), (115, 0.07), (150, 0.035)):
        delay = round(delay_ms / 1000 * SR)
        dest = dry_end + delay
        take = min(len(source), count - dest, max(0, tail - delay))
        if take > 0:
            result[dest:dest + take] += source[:take] * level * np.linspace(1, 0, take)[:, None]
    fade = min(round(0.015 * SR), count // 4)
    if fade:
        result[dry_end - fade:dry_end] *= np.linspace(1, 0, fade)[:, None]
        result[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return result, {"method": "strong_late_onset_with_bounded_tail",
                    "onset_sec": onset / SR, "dry_end_sec": dry_end / SR, "tail_ms": tail_ms,
                    "target_onset_sec": target}


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


def duck(music: np.ndarray, dialogue: np.ndarray | None, cue: MusicCue,
         previous_cutoff: int | None = None) -> tuple[np.ndarray, dict]:
    cutoffs = {"barely_behind": 12000, "background": 9500,
               "cinematic_distant": 6500, "montage_foreground": 16000}
    filtered = sosfilt(butter(2, 45, "highpass", fs=SR, output="sos"), music, axis=0)
    cutoff = cue.lowpass_hz or cutoffs[cue.depth_profile]
    highpassed = filtered
    filtered = sosfilt(butter(2, cutoff, "lowpass", fs=SR, output="sos"), highpassed, axis=0)
    if previous_cutoff and previous_cutoff != cutoff:
        length = min(len(music), round(0.2 * SR))
        old = sosfilt(butter(2, previous_cutoff, "lowpass", fs=SR, output="sos"), highpassed[:length], axis=0)
        blend = np.linspace(0, 1, length)[:, None]
        filtered[:length] = old * (1 - blend) + filtered[:length] * blend
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


def build_stems(sheet: SpottingSheet, store, dialogue: Path | None, root: Path,
                config: RunConfig | None = None) -> tuple[Path, Path, list]:
    bed_path, sfx_path = root / "music.wav", root / "sfx.wav"
    records, cursor = [], 0
    previous_source, phase_origin, previous_cutoff = None, 0, None
    lookup = {c.cue_id: c for c in sheet.cues}
    with wave.open(str(bed_path), "wb") as bed, wave.open(str(sfx_path), "wb") as sfx:
        for w in (bed, sfx):
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
        for cue in sheet.cues:
            start, end = round(cue.start_sec * SR), round(cue.end_sec * SR)
            gap = start - cursor
            if gap > 0:
                previous_source, previous_cutoff = None, None
            while gap > 0:
                block = min(gap, SR * 10)
                for w in (bed, sfx):
                    _write_frames(w, np.zeros((block, 2), np.float32))
                gap -= block
            source_id = cue.cue_id
            while lookup[source_id].reuse_music_from:
                source_id = lookup[source_id].reuse_music_from
            source = getattr(store, "safe_music", {}).get(source_id)
            if fully_silent(cue.start_sec, cue.end_sec, sheet.silence_windows):
                silence = np.zeros((end - start, 2), np.float32)
                write_audio(root / "stems" / f"{cue.cue_id}_prepared.wav", silence)
                _write_frames(bed, silence)
                _write_frames(sfx, silence)
                cursor, previous_source, previous_cutoff = end, None, None
                records.append({"cue_id": cue.cue_id, "entry_offset_sec": 0,
                                "analysis": {"onsets": [], "beats": [], "downbeat_verified": False, "method": "protected_silence"},
                                "control": {"gap_control": "protected_silence"},
                                "speech_band_music_to_voice_db": None})
                continue
            if source is None:
                if config is None or config.vocal_policy != "off":
                    raise PipelineError(f"Missing vocal-gate approval for {source_id}")
                source = root / "stems" / f"{source_id}_music.wav"
            analysis = onset_analysis(source)
            raw, _ = read_audio(source)
            manual_downbeat = store.assets["downbeats"].get(source_id)
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
            loop = plan_loop(raw, analysis, entry, store.assets.get("loops", {}).get(source_id),
                             config.beats_per_bar if config else 4)
            source_identity = (store.assets["music"].get(source_id, source),
                               loop["start_sec"], loop["end_sec"])
            if source_identity != previous_source:
                phase_origin = cue.start_sec
            phase = cue.start_sec - phase_origin
            music = fit_music(raw, seconds, entry, region=loop, phase_sec=phase,
                              crossfade_ms=config.loop_crossfade_ms if config else 40)
            timeline_onsets = []
            period = loop["period_sec"]
            for onset in analysis["onsets"]:
                if loop["start_sec"] <= onset < loop["end_sec"]:
                    time = (onset - loop["start_sec"] - phase) % period
                    while time < seconds:
                        timeline_onsets.append(cue.start_sec + time)
                        time += period
            exit_record = None
            if cue.exit_type == "transient_stop":
                music, exit_record = transient_exit(music, cue.exit_tail_ms, cue.exit_anchor_before_sec)
            voice = read_dialogue_window(dialogue, cue.start_sec, end - start) if dialogue else None
            processed, metrics = duck(music, voice, cue, previous_cutoff)
            previous_source = source_identity
            previous_cutoff = cue.lowpass_hz or {"barely_behind": 12000, "background": 9500,
                                                "cinematic_distant": 6500, "montage_foreground": 16000}[cue.depth_profile]
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
                            "loop": loop, "loop_phase_sec": phase, "exit": exit_record,
                            "timeline_onsets_sec": sorted(timeline_onsets),
                            "music_source_cue_id": source_id,
                            "lowpass_hz": previous_cutoff, "control": metrics,
                            "speech_band_music_to_voice_db": overlap_db})
        tail = round(sheet.video_duration_sec * SR) - cursor
        while tail > 0:
            count = min(tail, SR * 10)
            for w in (bed, sfx):
                _write_frames(w, np.zeros((count, 2), np.float32))
            tail -= count
    return bed_path, sfx_path, records


def render(dialogue: Path, music: Path, sfx: Path, root: Path, config: RunConfig, pacing=None, visual_source=None) -> dict:
    premix, master = root / "premix.wav", root / "master.wav"
    ffmpeg("-i", dialogue, "-i", music, "-i", sfx,
           "-filter_complex", "[0:a]highpass=f=65:p=2[v];[v][1:a][2:a]amix=inputs=3:duration=first:normalize=0[m]",
           "-map", "[m]", "-ar", SR, "-c:a", "pcm_f32le", premix)
    normalize(premix, master, config.target_lufs, config.true_peak_db)
    master_qc = loudness(master)
    video = root / f"final_video_{root.name}.mp4"
    # Re-encode for predictable MP4 compatibility, preserving image dimensions and frame rate.
    video_filter = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    if visual_source and pacing and pacing.visuals:
        from .visual_pacing import ffmpeg_filter
        video_filter = ffmpeg_filter(pacing, visual_source)
    ffmpeg("-i", config.video_path, "-i", master, "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-vf", video_filter, "-c:a", "aac", "-b:a", "192k",
           "-ar", SR, "-movflags", "+faststart", "-shortest", video)
    # Measure the actual encoded output: AAC can change true peaks.
    final_qc = loudness(video)
    actual_i, actual_tp = final_qc.get("input_i"), final_qc.get("input_tp")
    passed = (actual_i is not None and actual_tp is not None and
              abs(actual_i - config.target_lufs) <= 1 and actual_tp <= -1)
    visual_qc = None
    if visual_source and pacing and pacing.visuals:
        stream = next(s for s in probe(video)["streams"] if s["codec_type"] == "video")
        actual_frames = int(stream.get("nb_frames", -1))
        same_rate = Fraction(stream["avg_frame_rate"]) == Fraction(visual_source["rate"])
        visual_qc = {"source_frames": visual_source["frames"], "output_frames": actual_frames,
                     "frame_rate_preserved": same_rate,
                     "passed": actual_frames == visual_source["frames"] and same_rate}
        passed = passed and visual_qc["passed"]
    premix.unlink(missing_ok=True)
    return {"master": master.name, "video": video.name, "master_measurement": master_qc,
            "visual_pacing": "rendered" if visual_source and pacing and pacing.visuals else "source_picture",
            "visual_qc": visual_qc,
            "encoded_measurement": final_qc, "passed": passed,
            "target_lufs": config.target_lufs, "target_peak_db": config.true_peak_db,
            "qc_peak_ceiling_db": -1,
            "note": "QC failure requires review; a file is not upload-ready merely because it rendered."}
