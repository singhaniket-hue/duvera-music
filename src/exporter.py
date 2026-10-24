"""Human-readable cue sheets, CSV, FCP7 XML and optional OpenTimelineIO."""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .director import music_prompt
from .models import RunConfig, SpottingSheet, Transcript


def timestamp(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes):02}:{sec:05.2f}"


def export_script(sheet: SpottingSheet, transcript: Transcript, config: RunConfig, root: Path):
    lines = ["# Annotated music production script", "",
             f"Language: {config.language} · timing: **{sheet.timing_source}** · planner: {sheet.planner}",
             f"Master preset: {config.target_lufs} LUFS; {config.true_peak_db} dBTP. These are editable production targets.", ""]
    if config.sfx_silence_policy == "accents":
        lines += ["**Explicit exception:** editorial accents may play inside the MUSIC/SFX SILENCE windows below. "
                  "Music, ambience and Foley remain protected. See pacing_plan.json.", ""]
    lines += [f"> {warning}" for warning in sheet.warnings]
    for cue in sheet.cues:
        lines += ["", f"## {cue.cue_id} — {timestamp(cue.start_sec)}–{timestamp(cue.end_sec)}",
                  f"{cue.narrative_purpose}. Mood: {', '.join(cue.mood_descriptors)}.", "",
                  f"Cadence estimate: {cue.speech_syllables_per_sec:.2f} syllables/sec. "
                  f"Tempo suggestion: **{cue.target_bpm} BPM** (half/double-time alternatives may fit).",
                  f"Dialogue/music gap target: {cue.target_lu_gap} LU; depth: {cue.depth_profile}; exit: {cue.exit_type}.",
                  f"Music LPF override: {cue.lowpass_hz or 'depth default'} Hz; bounded exit tail: {cue.exit_tail_ms} ms.",
                  f"Music source: {cue.reuse_music_from or 'own cue'}; "
                  f"pre-silence transient target: {cue.exit_anchor_before_sec or 'automatic'} sec before boundary.",
                  "Mix: music HPF 45 Hz; depth LPF; center presence reduction up to 3.5 dB at 1.5–4 kHz "
                  "during speech; bounded 12 dB momentary-loudness ducking; 15 ms exit fade.",
                  "Entry: use a supplied downbeat offset when available; automatic onset entry is not a verified downbeat.",
                  "", "Generation prompt:", "", music_prompt(cue), "", "Spoken text:", ""]
        for segment in transcript.segments:
            if segment.start < cue.end_sec and segment.end > cue.start_sec:
                lines.append(f"> [{timestamp(segment.start)}] {segment.text}")
        for silence in sheet.silence_windows:
            if silence.start_sec < cue.end_sec and silence.end_sec > cue.start_sec:
                lines += ["", f"**MUSIC/SFX SILENCE {timestamp(silence.start_sec)}–{timestamp(silence.end_sec)}:** "
                          f"{silence.reason}. Dialogue continues."]
    (root / "annotated_music_script.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (root / "cue_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cue", "start_sec", "end_sec", "bpm", "gap_lu", "purpose", "prompt"])
        for cue in sheet.cues:
            writer.writerow([cue.cue_id, cue.start_sec, cue.end_sec, cue.target_bpm,
                             cue.target_lu_gap, cue.narrative_purpose, music_prompt(cue)])


def export_timeline(sheet: SpottingSheet, config: RunConfig, root: Path, records: list, dialogue: Path | None, pacing=None):
    fps, total = config.fps, round(sheet.video_duration_sec * config.fps)
    xml = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(xml, "sequence", id="duvera-sequence")
    ET.SubElement(sequence, "name").text = "Duvera Music Director"
    ET.SubElement(sequence, "duration").text = str(total)
    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = str(fps)
    ET.SubElement(rate, "ntsc").text = "FALSE"
    media = ET.SubElement(sequence, "media")
    audio = ET.SubElement(media, "audio")

    def clip(track, path, name, start, end, channel=1, video=False):
        element = ET.SubElement(track, "clipitem", id=f"{name}-{channel}")
        ET.SubElement(element, "name").text = name
        for key, value in {"start": round(start * fps), "end": round(end * fps), "in": 0,
                           "out": round((end - start) * fps), "duration": round((end - start) * fps)}.items():
            ET.SubElement(element, key).text = str(value)
        element.append(ET.fromstring(ET.tostring(rate)))
        file = ET.SubElement(element, "file", id=f"file-{name}-{channel}")
        ET.SubElement(file, "name").text = path.name
        ET.SubElement(file, "pathurl").text = path.resolve().as_uri()
        file.append(ET.fromstring(ET.tostring(rate)))
        fm = ET.SubElement(file, "media")
        fa = ET.SubElement(fm, "video" if video else "audio")
        if not video:
            ET.SubElement(fa, "channelcount").text = "2"
            source = ET.SubElement(element, "sourcetrack")
            ET.SubElement(source, "mediatype").text = "audio"
            ET.SubElement(source, "trackindex").text = str(channel)

    if config.video_path:
        track = ET.SubElement(ET.SubElement(media, "video"), "track")
        clip(track, config.video_path, "source-video", 0, sheet.video_duration_sec, video=True)
    for channel in (1, 2):
        if dialogue:
            clip(ET.SubElement(audio, "track"), dialogue, "dialogue", 0, sheet.video_duration_sec, channel)
        track = ET.SubElement(audio, "track")
        for cue in sheet.cues:
            clip(track, root / "stems" / f"{cue.cue_id}_prepared.wav", cue.cue_id,
                 cue.start_sec, cue.end_sec, channel)
        effect_files = ["ambience", "foley", "accent", "bridge_sfx"] if (root / "ambience.wav").exists() else ["sfx"]
        for name in effect_files:
            clip(ET.SubElement(audio, "track"), root / f"{name}.wav", name, 0, sheet.video_duration_sec, channel)
    markers = []
    if pacing:
        for effect in pacing.effects:
            markers.append({"time": effect.visual_time_sec, "name": effect.tier + ": " + effect.category,
                            "comment": effect.event_id + "; sample placement is baked into SFX WAV, not XML frames"})
        for visual in pacing.visuals:
            markers.append({"time": visual.start_sec, "name": "Eye-anchored punch-in",
                            "comment": f"{visual.zoom_to}x to {visual.end_sec}s; see pacing_plan.json. NLE crop not baked."})
    for event in sheet.visual_events:
        markers.append({"time": event.start_sec, "name": "Visual black boundary",
                        "comment": event.kind + "; luminance evidence, not OCR"})
    for silence in sheet.silence_windows:
        markers.append({"time": silence.start_sec, "name": "Music silence", "comment": silence.reason})
    for cue, record in zip(sheet.cues, records):
        markers.append({"time": cue.start_sec, "name": cue.cue_id, "comment": cue.narrative_purpose})
        last = -10
        onsets = record.get("timeline_onsets_sec", [cue.start_sec + x - record["entry_offset_sec"]
                                                   for x in record["analysis"]["onsets"]])
        for time in onsets:
            if cue.start_sec <= time < cue.end_sec and time - last >= 2:
                markers.append({"time": time, "name": "Suggested visual accent", "comment": "Onset; not verified downbeat"})
                last = time
    for marker in markers:
        m = ET.SubElement(sequence, "marker")
        for key, value in {"name": marker["name"], "comment": marker["comment"],
                           "in": str(round(marker["time"] * fps)), "out": "-1"}.items():
            ET.SubElement(m, key).text = value
    ET.indent(xml)
    ET.ElementTree(xml).write(root / "timeline.xml", encoding="utf-8", xml_declaration=True)
    (root / "markers.json").write_text(json.dumps(markers, indent=2), encoding="utf-8")
    try:
        import opentimelineio as otio
    except ImportError:
        return
    timeline = otio.schema.Timeline(name="Duvera Music Director")
    effect_tracks = [(name, root / f"{name}.wav") for name in effect_files]
    for name, path in [("Dialogue", dialogue), ("Music", root / "music.wav"), *effect_tracks]:
        if path is None:
            continue
        track = otio.schema.Track(name=name, kind=otio.schema.TrackKind.Audio)
        reference = otio.schema.ExternalReference(target_url=path.resolve().as_uri())
        track.append(otio.schema.Clip(name=name, media_reference=reference,
                     source_range=otio.opentime.TimeRange(otio.opentime.RationalTime(0, fps),
                                  otio.opentime.RationalTime(total, fps))))
        timeline.tracks.append(track)
    otio.adapters.write_to_file(timeline, str(root / "timeline.otio"))
