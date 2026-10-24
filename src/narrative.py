"""Conservative narrative guards shared by rule and LLM planners."""
from __future__ import annotations

import re
import unicodedata

from .models import SilenceWindow, SpottingSheet, Transcript, VisualEvent

# Contrast alone changes color; it does not automatically mute every 'but' or 'lekin'.
CONTRAST = ("par pata nahi kyun", "par pata nahin kyun", "didn't enjoy", "did not enjoy",
            "lekin", "लेकिन", "पर पता नहीं", "खाना पसंद नहीं")
REFLECTIVE = ("soch mein", "soch me ", "sochne laga", "thinking on the way", "i wondered",
              "shayad dikkat", "शायद दिक्कत", "सोच में", "सोचने लगा")
TWIST = ("pet kharab", "stomach upset", "finally realize", "finally realise", "finally realized",
         "hajam nahi", "hazam nahi", "jis tarah kutte", "पेट खराब", "हजम नहीं", "हज़म नहीं",
         "असली बात", "but then i realized", "asliyat mein", "असलियत में")
RESUME = ("agar aap bhi", "if you're like me", "if you are like me", "subscribe",
          "अगली बार", "अगर आप भी")
EMPHASIS = ("the truth", "i'm fine", "i was afraid", "sach", "asli problem", "aap sochiye", "सच")


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[,!?।.]", " ", text)).strip()


def narrative_events(transcript: Transcript) -> list[dict]:
    events = []
    for segment in transcript.segments:
        text = normalized(segment.text)
        found = []
        for kind, phrases in (("resume", RESUME), ("twist", TWIST), ("reflective", REFLECTIVE),
                              ("contrast", CONTRAST), ("emphasis", EMPHASIS)):
            matches = [phrase for phrase in phrases if phrase in text]
            if matches:
                start, end, anchor = segment.start, segment.end, "segment"
                words = [w for w in transcript.words if segment.start <= w.start < segment.end]
                for i, word in enumerate(words):
                    fragment = normalized(" ".join(w.word for w in words[i:i + 8]))
                    if any(fragment.startswith(phrase) for phrase in matches):
                        start, anchor = word.start, "word"
                        break
                found.append({"start": start, "end": end, "kind": kind, "anchor": anchor,
                              "text_offset": min(text.index(phrase) for phrase in matches)})
        # Several word-anchored turns may occur inside one ASR chunk. Without words, retain
        # the earliest textual turn rather than pretending to know later timestamps.
        anchored = [e for e in found if e["anchor"] == "word"]
        events.extend(anchored if anchored else sorted(found, key=lambda e: e["text_offset"])[:1])
    return sorted(events, key=lambda e: (e["start"], e["text_offset"]))


def fully_silent(start: float, end: float, windows: list[SilenceWindow]) -> bool:
    cursor = start
    for window in sorted(windows, key=lambda s: s.start_sec):
        if window.start_sec > cursor + 0.001:
            break
        if window.end_sec > cursor:
            cursor = window.end_sec
        if cursor >= end - 0.001:
            return True
    return False


def protect_narrative(sheet: SpottingSheet, transcript: Transcript,
                      visuals: list[VisualEvent] | None = None) -> SpottingSheet:
    result = sheet.model_copy(deep=True)
    result.schema_version = "1.1"
    result.visual_events = visuals or []
    events = narrative_events(transcript)
    resumes = sorted(e["start"] for e in events if e["kind"] == "resume")
    windows = list(result.silence_windows)
    for event in events:
        if event["kind"] == "twist":
            end = min(result.video_duration_sec, max(event["start"] + 3, event["end"]), event["start"] + 10)
            next_resume = next((x for x in resumes if x > event["start"]), None)
            if next_resume is not None:
                end = min(end, next_resume)
            if end > event["start"]:
                windows.append(SilenceWindow(start_sec=max(0, event["start"] - 0.15), end_sec=end,
                                              reason="Transcript twist/punchline; review segment-level timing"))
    for visual in result.visual_events:
        end = visual.end_sec if visual.kind == "black_card_candidate" else min(visual.end_sec, visual.start_sec + 3)
        # The CTA may still be on a black card: transcript resumption takes priority over darkness.
        resume = next((x for x in resumes if visual.start_sec < x < end), None)
        if resume is not None:
            end = resume
        windows.append(SilenceWindow(start_sec=visual.start_sec, end_sec=end,
                                      reason="Abrupt sustained black-card boundary; music/SFX stop, dialogue continues"))
    # Merge adjacent or overlapping protections; use exact protected endpoints.
    merged = []
    for window in sorted(windows, key=lambda s: s.start_sec):
        if merged and window.start_sec <= merged[-1].end_sec + 0.001:
            merged[-1].end_sec = max(merged[-1].end_sec, window.end_sec)
            if window.reason not in merged[-1].reason:
                merged[-1].reason += "; " + window.reason
        else:
            merged.append(window.model_copy())
    result.silence_windows = merged
    boundaries = {c.start_sec for c in result.cues} | {c.end_sec for c in result.cues}
    boundaries |= {e["start"] for e in events if e["kind"] in ("contrast", "reflective", "twist", "resume")}
    boundaries |= {x for w in merged for x in (w.start_sec, w.end_sec)}
    boundaries |= {v.start_sec for v in result.visual_events}
    cues = []
    for original in result.cues:
        points = sorted({original.start_sec, original.end_sec} |
                        {x for x in boundaries if original.start_sec < x < original.end_sec})
        for start, end in zip(points, points[1:]):
            cue = original.model_copy(deep=True)
            cue.start_sec, cue.end_sec = start, end
            # A split clip is not a newly generated transition. Preserve authored SFX only at the old end.
            if end < original.end_sec - 0.001:
                cue.bridge_sfx_prompt = None
                cue.exit_type = "crossfade"
            recent = [e for e in events if e["start"] <= start + 0.001 and e["kind"] != "emphasis"]
            state = recent[-1]["kind"] if recent else None
            if state in ("reflective", "contrast"):
                cue.mood_descriptors = ["pensive", "quirky", "restrained"]
                cue.narrative_purpose = "Reflective turn" if state == "reflective" else "Skeptical contrast"
                cue.depth_profile = "cinematic_distant"
                cue.lowpass_hz = 4500 if state == "reflective" else 6500
            elif state == "resume":
                cue.mood_descriptors = ["warm", "light", "resolved"]
                cue.narrative_purpose = "Return after punchline / call to action"
                cue.depth_profile, cue.lowpass_hz = "background", 9500
            if any(abs(end - w.start_sec) < 0.001 for w in merged):
                cue.exit_type, cue.exit_tail_ms, cue.bridge_sfx_prompt = "transient_stop", 200, None
            if fully_silent(start, end, merged):
                cue.narrative_purpose = "Protected narrative silence (dialogue only)"
                cue.bridge_sfx_prompt, cue.exit_type, cue.exit_tail_ms = None, "crossfade", 0
            cue.cue_id = f"cue_{len(cues) + 1:03}"
            cues.append(cue)
    result.cues = cues
    result.warnings = list(dict.fromkeys(result.warnings + [
        "Narrative phrase anchors use word timestamps when supplied, otherwise segment starts; review intent.",
        "Black-card detection uses luminance and sparse highlights; it does not read or understand on-screen text.",
    ]))
    return SpottingSheet.model_validate(result.model_dump())
