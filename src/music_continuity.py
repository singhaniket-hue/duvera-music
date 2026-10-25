"""Detect silent asset intros and unplanned bed gaps; preserve reviewed locked bed samples."""
from __future__ import annotations

import hashlib
import shutil
import wave
from pathlib import Path

import numpy as np

from .media import PipelineError, SR, read_audio, write_audio
from .models import RunConfig, SpottingSheet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audible_entry(audio: np.ndarray) -> float:
    """Two consecutive 10ms blocks above a relative/absolute floor, not a beat estimate."""
    hop = SR // 100
    blocks = len(audio) // hop
    if not blocks:
        return 0.
    power = np.mean(audio[:blocks * hop].astype(np.float64).reshape(blocks, hop, 2) ** 2, axis=(1, 2))
    floor = max(10 ** (-65 / 10), float(power.max(initial=0)) * 1e-4)
    active = power > floor
    sustained = np.flatnonzero(active[:-1] & active[1:])
    if not len(sustained):
        raise PipelineError("Music asset has no sustained audible audio")
    return max(0, (int(sustained[0]) * hop - round(0.005 * SR)) / SR)


def validate_locked_music(config: RunConfig, duration: float, sheet: SpottingSheet):
    source = config.locked_music_path
    if sha256(source) != config.locked_music_sha256:
        raise PipelineError("Locked music SHA-256 mismatch; no provider was called")
    try:
        audio, rate = read_audio(source)
    except (wave.Error, EOFError) as exc:
        raise PipelineError("Locked music must be a PCM16 WAV music-only stem, not a muxed video/master") from exc
    if rate != SR or audio.shape[1] != 2 or len(audio) != round(duration * SR):
        raise PipelineError("Locked music must be a timeline-length 48kHz stereo PCM16 music-only WAV")
    for silence in sheet.silence_windows:
        region = audio[round(silence.start_sec * SR):round(silence.end_sec * SR)]
        if np.max(np.abs(region), initial=0) > 1 / 32768:
            raise PipelineError("Locked music conflicts with current silence windows; reuse its original spotting sheet")
    return audio


def build_locked_stems(sheet: SpottingSheet, config: RunConfig, root: Path):
    audio = validate_locked_music(config, sheet.video_duration_sec, sheet)
    bed, sfx = root / "music.wav", root / "sfx.wav"
    shutil.copyfile(config.locked_music_path, bed)
    if sha256(bed) != config.locked_music_sha256:
        raise PipelineError("Locked music changed while copying; no altered bed will be mixed")
    write_audio(sfx, np.zeros_like(audio))
    records = []
    for cue in sheet.cues:
        write_audio(root / "stems" / f"{cue.cue_id}_prepared.wav",
                    audio[round(cue.start_sec * SR):round(cue.end_sec * SR)])
        records.append({"cue_id": cue.cue_id, "entry_offset_sec": 0,
            "analysis": {"onsets": [], "beats": [], "downbeat_verified": False, "method": "locked_timeline_bed"},
            "control": {"gap_control": "locked_samples_unchanged"}, "speech_band_music_to_voice_db": None,
            "locked_sha256": config.locked_music_sha256})
    return bed, sfx, records


def music_gap_qc(path: Path, sheet: SpottingSheet, max_gap: float) -> dict:
    """A diagnostic threshold, not an artistic judgement: declare intentional gaps in the sheet."""
    hop = round(0.01 * SR)
    quiet, times = [], []
    with wave.open(str(path), "rb") as reader:
        position = 0
        while raw := reader.readframes(hop):
            audio = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
            a, b = position / SR, (position + len(audio) // 2) / SR
            # Permit the bounded transient tail just before a protected silence.
            intentional = any(a < s.end_sec and b > max(0, s.start_sec - 0.6) for s in sheet.silence_windows)
            quiet.append(not intentional and float(np.mean(audio ** 2)) < 10 ** (-60 / 10))
            times.append(a)
            position += len(audio) // 2
    indices = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
    gaps = [{"start_sec": times[a], "end_sec": min(position / SR, b * hop / SR)}
            for a, b in indices.reshape(-1, 2) if (b - a) * hop / SR > max_gap]
    return {"passed": not gaps, "unexpected_gaps": gaps, "threshold_dbfs": -60,
            "max_gap_sec": max_gap, "note": "Intentional pauses must be marked as SilenceWindow; not a musicality score"}
