"""Explicit archival windows or opt-in reference-voice candidates; no identity claims."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import Field, model_validator
from scipy.signal import resample_poly

from .media import PipelineError, SR, extract_audio, read_audio
from .models import Contract, SilenceWindow, SpottingSheet


class ArchivalInsertWindow(Contract):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    confidence: float = Field(default=1, ge=0, le=1)
    evidence: str
    source: str = "reviewed"

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("Archival window end must follow start")
        return self


class ArchivalWindows(Contract):
    windows: list[ArchivalInsertWindow] = Field(default_factory=list)


def mismatch_windows(scores: list[dict], duration: float, low: float, high: float) -> list[ArchivalInsertWindow]:
    """Two low windows enter a candidate; two high windows resume. Low speech does not vote."""
    result, pending, returning = [], [], []
    opened = None
    for score in scores:
        value = score.get("cosine")
        if value is None:
            pending, returning = [], []
            continue
        if opened is None:
            pending = pending + [score] if value < low else []
            if len(pending) >= 2:
                opened = pending[0]["start_sec"]
                pending = []
        else:
            returning = returning + [score] if value >= high else []
            if len(returning) >= 2:
                end = max(opened + 0.01, returning[0]["start_sec"])
                result.append(ArchivalInsertWindow(start_sec=opened, end_sec=min(duration, end),
                    confidence=0.7, source="reference_voice_candidate",
                    evidence="Persistent low narrator cosine followed by stable return; not proof of archival origin"))
                opened, returning = None, []
    if opened is not None and opened < duration:
        result.append(ArchivalInsertWindow(start_sec=opened, end_sec=duration, confidence=0.7,
            source="reference_voice_candidate", evidence="Persistent mismatch; no stable narrator return detected"))
    return result


def detect_reference_windows(dialogue: Path, reference: Path, root: Path, config):
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as exc:
        raise PipelineError("Reference-voice analysis requires the speaker extra; reviewed archival windows need no model") from exc
    try:
        encoder = VoiceEncoder(device="cpu", verbose=False)
    except Exception as exc:
        raise PipelineError("Reference-voice model could not load; check the optional speaker installation/weights") from exc
    ref_file = root / "narrator_reference_analysis.wav"
    extract_audio(reference, ref_file)
    ref, _ = read_audio(ref_file)
    ref16 = preprocess_wav(resample_poly(ref.mean(axis=1), 1, 3), source_sr=16000)
    if len(ref16) < 3 * 16000:
        raise PipelineError("Narrator reference needs at least three seconds of clean detected speech")
    reference_embedding = encoder.embed_utterance(ref16)
    if not np.isfinite(reference_embedding).all():
        raise PipelineError("Reference-voice model returned an invalid narrator embedding")
    audio, rate = read_audio(dialogue)
    if rate != SR:
        raise PipelineError("Speaker analysis source must be 48kHz")
    scores = []
    for start in range(0, max(1, len(audio) - 2 * SR + 1), SR // 2):
        clip = audio[start:start + 2 * SR]
        speech = preprocess_wav(resample_poly(clip.mean(axis=1), 1, 3), source_sr=16000)
        score = None
        if len(speech) >= 12000:
            vector = encoder.embed_utterance(speech)
            if not np.isfinite(vector).all():
                raise PipelineError("Reference-voice model returned invalid source embeddings")
            score = float(np.clip(np.dot(vector, reference_embedding) /
                max(np.linalg.norm(vector) * np.linalg.norm(reference_embedding), 1e-12), -1, 1))
        scores.append({"start_sec": start / SR, "end_sec": min(len(audio), start + 2 * SR) / SR,
                       "cosine": score})
    # No embeddings/voiceprints are persisted; only candidate timing/evidence scores.
    return mismatch_windows(scores, len(audio) / SR, config.speaker_mismatch_threshold,
                            config.speaker_return_threshold), scores


def archival_windows(config, dialogue: Path | None, duration: float, root: Path):
    windows, scores = [], []
    if config.archival_windows_path:
        windows += ArchivalWindows.model_validate_json(config.archival_windows_path.read_text(encoding="utf-8")).windows
    if config.archival_detection == "reference_voice":
        if dialogue is None:
            raise PipelineError("Reference-voice analysis requires recorded source audio")
        detected, scores = detect_reference_windows(dialogue, config.narrator_reference_path, root, config)
        windows += detected
    for window in windows:
        if window.end_sec > duration + 0.001:
            raise PipelineError("Archival insert exceeds source duration")
    return sorted(windows, key=lambda w: w.start_sec), scores


def protect_archival(sheet: SpottingSheet, windows: list[ArchivalInsertWindow]) -> SpottingSheet:
    result = sheet.model_copy(deep=True)
    for window in windows:
        if not any(s.kind == "archival" and s.start_sec == window.start_sec and s.end_sec == window.end_sec
                   for s in result.silence_windows):
            result.silence_windows.append(SilenceWindow(start_sec=window.start_sec, end_sec=window.end_sec,
                kind="archival", fade_sec=0.15, reason=f"Archival protection ({window.source}): {window.evidence}"))
    result.silence_windows.sort(key=lambda s: s.start_sec)
    return SpottingSheet.model_validate(result.model_dump())
