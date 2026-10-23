"""Media ingestion, optional local ASR, and explicit estimated script timing."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, periodogram

from .linguistics import TOKEN, whisper_options
from .media import PipelineError, SR, extract_audio, probe, read_audio
from .models import RunConfig, Segment, Transcript


def estimated_transcript(text: str, wpm: float) -> Transcript:
    lines = [x.strip() for x in re.split(r"(?<=[.!?।])\s+|\n+", text) if x.strip()]
    cursor, segments = 0.0, []
    for line in lines:
        duration = max(1.0, len(TOKEN.findall(line)) * 60 / wpm)
        segments.append(Segment(start=cursor, end=cursor + duration, text=line))
        cursor += duration
    return Transcript(segments=segments, timing_source="estimated")


def transcribe(path: Path, config: RunConfig, hint: str) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PipelineError("Media without a supplied transcript requires pip install '.[asr]'. "
                            "Alternatively supply --transcript with actual timestamps.") from exc
    model = WhisperModel(config.asr_model, device=config.asr_device,
                         compute_type="int8" if config.asr_device == "cpu" else "float16")
    chunks, _ = model.transcribe(str(path), **whisper_options(config.language, hint))
    segments, words = [], []
    for s in chunks:
        if s.text.strip() and s.end > s.start:
            segments.append(Segment(start=max(0, s.start), end=s.end, text=s.text.strip()))
            words.extend({"start": w.start, "end": w.end, "word": w.word} for w in (s.words or []))
    (config.output_dir / "word_timestamps.json").write_text(json.dumps(words, ensure_ascii=False, indent=2))
    return Transcript(segments=segments, timing_source="asr")


def ingest(config: RunConfig) -> tuple[Transcript, float, Path | None]:
    hint = config.script_path.read_text(encoding="utf-8") if config.script_path else ""
    dialogue, duration = None, None
    if config.video_path:
        info = probe(config.video_path)
        if not any(s["codec_type"] == "audio" for s in info["streams"]):
            raise PipelineError("Input media has no dialogue audio stream")
        if config.mode == "final_video" and not any(s["codec_type"] == "video" for s in info["streams"]):
            raise PipelineError("final_video requires a video stream; use assets_and_script for audio inputs")
        dialogue = config.output_dir / "dialogue.wav"
        videos = [s for s in info["streams"] if s["codec_type"] == "video"]
        video_duration = float(videos[0].get("duration") or info["format"]["duration"]) if videos else None
        # AAC decoders may expose padding beyond the video clock; do not extend the timeline with it.
        extract_audio(config.video_path, dialogue, video_duration)
        import wave
        with wave.open(str(dialogue)) as w:
            duration = w.getnframes() / w.getframerate()
    if config.transcript_path:
        raw = json.loads(config.transcript_path.read_text(encoding="utf-8"))
        transcript = Transcript.model_validate(raw)
    elif dialogue:
        transcript = transcribe(dialogue, config, hint)
    else:
        transcript = estimated_transcript(hint, config.estimated_wpm)
    transcript_end = transcript.segments[-1].end
    if duration and transcript_end > duration + 0.05:
        raise PipelineError(f"Transcript ends at {transcript_end:.3f}s, beyond audio duration {duration:.3f}s")
    (config.output_dir / "transcript.json").write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return transcript, duration or transcript_end, dialogue


def prosodic_pulse(samples: np.ndarray, sr: int = SR) -> float | None:
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    hop = max(1, int(sr / 100))
    n = len(mono) // hop
    if n < 200 or np.max(np.abs(mono), initial=0) < 1e-5:
        return None
    env = np.sqrt(np.mean(mono[:n * hop].reshape(n, hop) ** 2, axis=1))
    f, p = periodogram(env - env.mean(), fs=sr / hop)
    mask = (f >= 0.8) & (f <= 3)
    if not mask.any() or p[mask].max() < 1e-10:
        return None
    return float(f[mask][np.argmax(p[mask])])


def onset_analysis(path: Path, use_librosa: bool = True) -> dict:
    """Onset/beat estimates, explicitly never labeled true bar downbeats."""
    audio, sr = read_audio(path)
    mono = audio.mean(axis=1)
    if use_librosa:
        try:
            import librosa
            env = librosa.onset.onset_strength(y=mono, sr=sr)
            times = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="time")
            tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, units="time")
            return {"onsets": times.tolist(), "beats": beats.tolist(),
                    "estimated_bpm": float(np.asarray(tempo).ravel()[0]), "method": "librosa_beats",
                    "downbeat_verified": False}
        except ImportError:
            pass
    hop = 480
    n = len(mono) // hop
    if n < 3:
        return {"onsets": [], "beats": [], "estimated_bpm": None,
                "method": "rms_flux", "downbeat_verified": False}
    rms = np.sqrt(np.mean(mono[:n * hop].reshape(n, hop) ** 2, axis=1))
    flux = np.maximum(0, np.diff(rms, prepend=rms[0]))
    peaks, _ = find_peaks(flux, distance=15, prominence=max(float(flux.max()) * 0.12, 1e-5))
    return {"onsets": (peaks * hop / sr).tolist(), "beats": [], "estimated_bpm": None,
            "method": "rms_flux", "downbeat_verified": False}
