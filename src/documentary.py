"""Chapter variation for commentary; deterministic chapter labels remain editorial suggestions."""
from __future__ import annotations

import math

import numpy as np

from .linguistics import cadence
from .models import MusicCue, SilenceWindow, SpottingSheet
from .narrative import fully_silent
from .pacing_models import EditEvent

PALETTE = [
    ("Hook and thesis", ["tense", "investigative"], ["low synth pulse", "muted marimba"], 88),
    ("Cases and escalation", ["dark", "driving"], ["muted plucks", "restrained percussion"], 98),
    ("History and analysis", ["curious", "analytical"], ["sparse piano", "muted strings"], 86),
    ("Psychology and evidence", ["minimal", "cerebral"], ["sparse sub pulse", "soft string texture"], 76),
    ("Synthesis and reflection", ["warm", "reflective"], ["warm piano", "gentle strings"], 80),
]


def documentary_baseline(transcript, duration, config, dialogue_lufs=None, pulse=None):
    # Five narrative acts fit 7.5–12.5 minutes. Other runtimes use a comparable span,
    # not hard-coded timestamps copied from another creator's video.
    count = 1 if duration <= 180 else 5 if 450 <= duration <= 750 else max(1, round(duration / 120))
    while duration > 180 and duration / count > 150:
        count += 1
    boundaries, warnings = [0.], []
    candidates = sorted({s.start for s in transcript.segments if s.start > 0})
    for index in range(1, count):
        target = duration * index / count
        lower = max(boundaries[-1] + 90, duration - (count - index) * 150)
        upper = min(boundaries[-1] + 150, duration - (count - index) * 90)
        allowed = [t for t in candidates if lower <= t <= upper]
        boundary = min(allowed, key=lambda t: abs(t - target)) if allowed else float(np.clip(target, lower, upper))
        if not allowed:
            warnings.append("Some chapter boundaries use a duration-grid fallback, not a detected semantic break")
        boundaries.append(boundary)
    boundaries.append(duration)
    cues = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        stage = round(index * 4 / max(1, count - 1)) if count > 1 else 0
        title, mood, instruments, bpm = PALETTE[stage]
        text = " ".join(s.text for s in transcript.segments if s.start < end and s.end > start)
        cues.append(MusicCue(cue_id=f"cue_{index + 1:03}", chapter_id=f"chapter_{index + 1:03}", chapter_title=title,
            start_sec=start, end_sec=end, narrative_purpose=f"Chapter {index + 1}: {title}",
            speech_syllables_per_sec=cadence(text, end - start, config.language), prosodic_pulse_hz=pulse,
            target_bpm=bpm, mood_descriptors=mood, allowed_instruments=instruments,
            depth_profile="background", target_lu_gap=15, generation_seconds=min(60, max(3, end - start))))
    silence = []
    for segment in transcript.segments:
        text = segment.text.casefold()
        if any(token in text for token in ("?", "mujhe batao", "aap batao", "आप बताइए", "what do you think")):
            start = segment.start if segment.end - segment.start <= 5 else max(segment.start, segment.end - 3)
            silence.append(SilenceWindow(start_sec=start, end_sec=min(segment.end, start + 5),
                reason="Suggested rhetorical-question hold; review segment-level timing"))
    last = transcript.segments[-1]
    start = max(last.start, duration - 12)
    if start < duration:
        silence.append(SilenceWindow(start_sec=start, end_sec=duration,
            reason="Commentary closing sentence: solo source voice; review actual sentence anchor"))
    warnings += ["Documentary chapter labels/moods and question holds are editorial proposals, not semantic ground truth",
                 "Loudness, chapter lengths and BPM are configurable production priors"]
    if transcript.timing_source == "estimated":
        warnings.append("Script timestamps are estimated from WPM and are NOT aligned to recorded speech.")
    return SpottingSheet(video_duration_sec=duration, dialogue_integrated_lufs=dialogue_lufs,
        timing_source=transcript.timing_source, planner="rules_documentary", cues=cues,
        silence_windows=silence, warnings=list(dict.fromkeys(warnings)))


def reuse_chapters(sheet: SpottingSheet) -> SpottingSheet:
    """Narrative splits reuse their chapter stem rather than requesting new songs per phrase."""
    result = sheet.model_copy(deep=True)
    first = {}
    for cue in result.cues:
        if cue.chapter_id and not fully_silent(cue.start_sec, cue.end_sec, result.silence_windows):
            cue.reuse_music_from = first.get(cue.chapter_id)
            first.setdefault(cue.chapter_id, cue.cue_id)
    return result


def chapter_summary(sheet):
    chapters = {}
    for cue in sheet.cues:
        if cue.chapter_id:
            entry = chapters.setdefault(cue.chapter_id, {"chapter_id": cue.chapter_id, "start_sec": cue.start_sec,
                "end_sec": cue.end_sec, "purpose": cue.chapter_title or cue.narrative_purpose, "mood": cue.mood_descriptors,
                "bpm": cue.target_bpm, "music_source_cue_id": None})
            entry["end_sec"] = max(entry["end_sec"], cue.end_sec)
            if not entry["music_source_cue_id"] and not fully_silent(cue.start_sec, cue.end_sec, sheet.silence_windows):
                entry["music_source_cue_id"] = cue.reuse_music_from or cue.cue_id
    return list(chapters.values())


def chapter_events(sheet):
    return [EditEvent(event_id=f"doc_{c['chapter_id']}", time_sec=c["start_sec"], kind="chapter_transition",
        confidence=0.9, intensity=0.9, semantic_salience=0.9,
        evidence="Planned documentary chapter boundary, not a detected visual event")
        for c in chapter_summary(sheet)[1:] if math.isfinite(c["start_sec"])]
