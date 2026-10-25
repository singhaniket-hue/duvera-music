"""Shared AV cuts, semantic proposals and deterministic remapping before the existing mixer."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from fractions import Fraction
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from .archival import ArchivalInsertWindow
from .documentary_schema import (DOCUMENTARY_DIRECTOR_PROMPT, DocumentaryEditPlan, KeepRange,
    PauseTrimInterval, TimelineMap, VisualEditEvent, reading_hold)
from .media import PipelineError, SR, read_audio, write_audio
from .models import Segment, SilenceWindow, SpottingSheet, Transcript, Word
from .pacing_models import EditEvent, EventGraph
from .visual_pacing import validate_render_source


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quiet_floor(audio):
    count = min(len(audio) // 960, 3000)
    if not count:
        return -60.
    power = np.mean(audio[:count * 960].astype(float).reshape(count, 960, 2) ** 2, axis=(1, 2))
    # Short recordings may contain less than 10% room tone. A speech-heavy percentile
    # would raise the deletion threshold enough to mistake a quiet breath for silence.
    return float(np.clip(10 * np.log10(max(float(np.percentile(power, 1)), 1e-12)), -90, -35))


def quiet_cut(audio, start, end):
    chunk = audio[round(start * SR):round(end * SR)]
    if not len(chunk):
        return False, {"reason": "empty audio"}
    rms = float(10 * np.log10(max(float(np.mean(chunk.astype(float) ** 2)), 1e-12)))
    peak = float(20 * np.log10(max(float(np.max(abs(chunk), initial=0)), 1e-12)))
    threshold = min(-38, quiet_floor(audio) + 6)
    return rms <= threshold and peak < -32, {"rms_dbfs": rms, "peak_dbfs": peak, "threshold_dbfs": threshold}


def pause_candidates(transcript, audio, inserts):
    if transcript.timing_source == "estimated" or audio is None:
        return [], [{"reason": "Automatic trims need recorded audio and aligned timestamps"}]
    speech = transcript.words or transcript.segments
    trims, decisions = [], []
    for left, right in zip(speech, speech[1:]):
        a, b = left.end, right.start
        if b - a <= 0.4:
            continue
        context = " ".join(s.text for s in transcript.segments if s.start < b + 0.2 and s.end > a - 0.2).casefold()
        dramatic = any(x in context for x in ("?", "kyun", "aakhir", "dark side", "sach", "why", "truth", "क्यों", "आखिर"))
        keep = 650 if dramatic else 320 if any(x in context for x in ("chapter", "now let's", "ab baat", "अध्याय")) else 180
        decision = {"raw_start_sec": a, "raw_end_sec": b, "keep_pause_ms": keep, "accepted": False}
        decisions.append(decision)
        if dramatic and b - a <= 0.9:
            decision["reason"] = "Dramatic/question pause preserved"
            continue
        if b - a <= keep / 1000 + 0.05 or any(a < w.end_sec and b > w.start_sec for w in inserts):
            decision["reason"] = "Already compact or archival audio protected"
            continue
        amount = b - a - keep / 1000
        centre = (a + b) / 2
        accepted, measurement = quiet_cut(audio, centre - amount / 2, centre + amount / 2)
        decision.update(measurement)
        if not accepted:
            decision["reason"] = "Middle contains audible activity/breath/noise above the quiet threshold"
            continue
        trims.append(PauseTrimInterval(raw_start_sec=a, raw_end_sec=b, keep_pause_ms=keep,
            classification="dramatic" if dramatic else "topic" if keep == 320 else "explanatory",
            evidence="Aligned speech gap plus low-energy middle; endpoint breaths retained"))
        decision.update(accepted=True, reason="Conservative quiet middle eligible for frame-aligned cut")
    return trims, decisions


def rules_edit_plan(transcript, duration, config, audio, inserts):
    from .director import rules_plan
    visuals, warnings = [], []
    last_graphic_end = -1.
    for segment in transcript.segments:
        text = segment.text.strip()
        lowered = text.casefold()
        structural = any(x in lowered for x in ("aakhir", "kyun", "why ", "क्यों", "आखिर")) and "?" in text
        statistic = bool(re.search(r"(?:[₹$]\s*\d|\d[\d,.]*\s*(?:%|percent|crore|lakh|million|billion|करोड़|प्रतिशत)|\b(?:19|20)\d{2}\b)", text, re.I))
        thesis = any(x in lowered for x in ("the truth", "dark side", "par asliyat", "lekin", "magar", "असलियत", "लेकिन"))
        if structural:
            kind = "black_chapter_card" if structural else "evidence_card" if statistic else "lower_third_citation"
            # Verbatim short excerpt; never fabricate an external source/date or verified citation.
            words = text.split()
            number = next((i for i, word in enumerate(words) if re.search(r"\d", word)), 0)
            headline = " ".join(words[max(0, number - 1):max(0, number - 1) + 6]).strip()
            source = None
            length = reading_hold(" ".join(x for x in (headline, source) if x)) + 0.3 + 1 / 24
            start = segment.start
            if start >= last_graphic_end + 0.1 and start + length <= duration:
                visuals.append(VisualEditEvent(start_sec=start, end_sec=start + length, edit_type=kind,
                    headline_text=headline, subtext_source=source,
                    sfx_trigger="sub_drop" if structural else "ui_data_tick" if statistic else "paper_slide_pop"))
                last_graphic_end = start + length
            else:
                warnings.append(f"Skipped overlapping/short reading interval near {start:.2f}s; review graphic proposal")
        if thesis and segment.start + 3 <= duration:
            if not any(e.edit_type.startswith("punch_in") and segment.start < e.end_sec for e in visuals):
                visuals.append(VisualEditEvent(start_sec=segment.start, end_sec=min(duration, segment.start + 4),
                    edit_type="punch_in_120_emphasis", sfx_trigger="none"))
    # Sparse eight-second framing changes; actual face/eye availability gates their execution.
    for start in np.arange(8, max(8, duration - 3), 8):
        if any(start < e.end_sec and start + 3 > e.start_sec for e in visuals):
            continue
        visuals.append(VisualEditEvent(start_sec=float(start), end_sec=min(duration, float(start) + 4),
            edit_type="punch_in_114", sfx_trigger="none"))
    trims, decisions = pause_candidates(transcript, audio, inserts)
    # Reading holds take priority over shortening a pause.
    trims = [p for p in trims if not any(e.is_graphic and p.raw_start_sec < e.end_sec and p.raw_end_sec > e.start_sec for e in visuals)]
    baseline = DocumentaryEditPlan(source_duration_sec=duration, visual_events=sorted(visuals, key=lambda e: e.start_sec),
        pause_trims=trims, music_sheet=rules_plan(transcript, duration, config), warnings=warnings + [
            "Evidence cards require a reviewed publisher source; generic narration cards are never generated",
            "Automatic eye crops execute only with actual eye geometry or a supplied reviewed anchor"])
    return baseline, decisions


def plan_edl(transcript, duration, config, audio, inserts):
    root = config.output_dir
    cached = root / "documentary_edit_plan.json"
    if cached.exists():
        return DocumentaryEditPlan.model_validate_json(cached.read_text(encoding="utf-8"))
    if config.documentary_plan_path:
        result = DocumentaryEditPlan.model_validate_json(config.documentary_plan_path.read_text(encoding="utf-8"))
    else:
        result, decisions = rules_edit_plan(transcript, duration, config, audio, inserts)
        save(root / "pause_analysis.json", {"decisions": decisions, "note": "Quiet-gap proxy, not a validated breath classifier"})
        if config.planner == "llm":
            url, model = os.getenv("DIRECTOR_LLM_URL"), os.getenv("DIRECTOR_LLM_MODEL")
            if not url or not model:
                raise PipelineError("Unified documentary planner needs the configured DIRECTOR_LLM_URL/MODEL")
            headers = {"Content-Type": "application/json"}
            if os.getenv("DIRECTOR_LLM_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["DIRECTOR_LLM_API_KEY"]
            payload = {"model": model, "temperature": 0.2, "messages": [
                {"role": "system", "content": DOCUMENTARY_DIRECTOR_PROMPT},
                {"role": "user", "content": json.dumps({"schema": DocumentaryEditPlan.model_json_schema(),
                    "baseline": result.model_dump(), "transcript": transcript.model_dump(),
                    "archival_windows": [w.model_dump() for w in inserts]}, ensure_ascii=False)}]}
            try:
                with urlopen(Request(url, data=json.dumps(payload).encode(), headers=headers), timeout=180) as response:
                    raw = json.load(response)["choices"][0]["message"]["content"].strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                result = DocumentaryEditPlan.model_validate_json(raw)
                if result.music_sheet is None:
                    raise ValueError("Unified LLM plan must include music_sheet")
                text = " ".join(s.text for s in transcript.segments)
                if any(e.eye_anchor is not None or (e.source_url and e.source_url not in text) for e in result.visual_events):
                    raise ValueError("Transcript-only LLM cannot invent measured eyes or source URLs")
            except Exception as exc:
                raise PipelineError("Unified documentary LLM/schema validation failed before generation") from exc
    if result.source_duration_sec is not None and abs(result.source_duration_sec - duration) > 0.05:
        raise PipelineError("Documentary EDL must match the source clock, not a previously edited render")
    result.source_duration_sec = duration
    result = DocumentaryEditPlan.model_validate(result.model_dump())
    if result.music_sheet and (abs(result.music_sheet.video_duration_sec - duration) > 0.05 or
                              result.music_sheet.timing_source != transcript.timing_source):
        raise PipelineError("EDL music sheet changed source duration/timing provenance")
    if result.music_sheet and config.preset == "social_commentary":
        from .documentary import chapter_summary, reuse_chapters
        result.music_sheet = reuse_chapters(result.music_sheet)
        if config.planner == "llm" and not config.documentary_plan_path and duration > 180:
            chapters = chapter_summary(result.music_sheet)
            if not chapters or any(not 89.99 <= c["end_sec"] - c["start_sec"] <= 150.01 for c in chapters):
                raise PipelineError("Unified LLM music must retain distinct 90–150s chapter spans")
    cached.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def build_time_map(plan, transcript, audio, inserts, source, retake_cuts=()):
    rate = Fraction(source["rate"])
    fps, frames = float(rate), source["frames"]
    cuts, decisions = [], []
    speech = transcript.words or transcript.segments
    for pause in plan.pause_trims:
        a, b = pause.raw_start_sec, pause.raw_end_sec
        amount = math.floor((b - a - pause.keep_pause_ms / 1000) * fps + 1e-7)
        if a < 0 or b > frames / fps + 0.05 or amount < 1:
            raise PipelineError("Pause trim exceeds source or removes less than one frame")
        middle = round((a + b) / 2 * fps)
        first = max(math.ceil(a * fps + 1e-7), middle - amount // 2)
        last = first + amount
        start, end = first / fps, last / fps
        if last > math.floor(b * fps - 1e-7):
            raise PipelineError("Frame-snapped trim does not fit its reviewed speech gap")
        if any(start < s.end - 0.002 and end > s.start + 0.002 for s in speech):
            raise PipelineError("Pause trim overlaps timestamped speech; words will not be deleted")
        if any(a < w.end_sec and b > w.start_sec for w in inserts):
            raise PipelineError("Pause trim overlaps a protected archival insert")
        if any(e.is_graphic and a < e.end_sec and b > e.start_sec for e in plan.visual_events):
            raise PipelineError("Pause trim overlaps a graphic reading interval")
        if audio is None or transcript.timing_source == "estimated":
            raise PipelineError("Pause trimming requires recorded audio and aligned timestamp provenance")
        quiet, measurement = quiet_cut(audio, start, end)
        decisions.append({"source_start_frame": first, "source_end_frame": last, **measurement})
        if not quiet:
            raise PipelineError("Pause trim contains audible activity; retain it or review a quieter interval")
        cuts.append((first, last))
    for cut in retake_cuts:
        first=max(0,round(cut['start_sec']*fps))
        last=min(frames,round(cut['end_sec']*fps))
        if last>first:
            cuts.append((first,last))
            decisions.append({**cut,'source_start_frame':first,'source_end_frame':last,
                              'accepted':True,'authority':'V2 source ASR retake/gap detector'})
    cuts.sort()
    merged=[]
    for a,b in cuts:
        if merged and a<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:
            merged.append((a,b))
    cuts=merged
    if any(b[0] < a[1] for a, b in zip(cuts, cuts[1:])):
        raise PipelineError("Frame-snapped cuts overlap")
    keep, cursor, output = [], 0, 0
    for a, b in cuts + [(frames, frames)]:
        if a > cursor:
            keep.append(KeepRange(source_start_frame=cursor, source_end_frame=a, output_start_frame=output))
            output += a - cursor
        cursor = b
    return TimelineMap(rate_num=rate.numerator, rate_den=rate.denominator, source_frames=frames, keep_ranges=keep), decisions


def edit_audio(audio, mapping, bridge=True):
    target = round(mapping.duration * SR)
    result = np.zeros((target, 2), np.float32)
    joins, rounding = [], 0
    for item in mapping.keep_ranges:
        a, b = round(item.source_start_frame / mapping.fps * SR), round(item.source_end_frame / mapping.fps * SR)
        out = round(item.output_start_frame / mapping.fps * SR)
        needed = round((item.output_start_frame + item.source_end_frame - item.source_start_frame) / mapping.fps * SR) - out
        chunk = audio[a:b]
        take = min(len(chunk), needed)
        result[out:out + take] = chunk[:take]
        if take < needed and take:
            result[out + take:out + needed] = chunk[-1]
        rounding += abs(len(chunk) - needed)
        joins.append((out, a, b))
    if bridge:
        half = round(0.01 * SR)
        for previous, following in zip(joins, joins[1:]):
            join, raw_in = following[0], following[1]
            raw_out = previous[2]
            if join < half or join + half > len(result) or raw_out + half > len(audio) or raw_in < half:
                continue
            left = audio[raw_out - half:raw_out + half]
            right = audio[raw_in - half:raw_in + half]
            phase = np.linspace(0, 1, 2 * half)
            result[join - half:join + half] = left * np.sqrt(1-phase)[:, None] + right * np.sqrt(phase)[:, None]
    if np.max(abs(result), initial=0) >= 1:
        raise PipelineError("Room-tone bridge would clip; source/edit needs review")
    return result, {"crossfade_ms": 20 if bridge else 0, "joins": max(0, len(joins) - 1),
                    "sample_rounding_adjustments": rounding, "output_samples": target}


def remap_transcript(transcript, mapping):
    segments=[]
    for s in transcript.segments:
        if mapping.time(s.end)<=mapping.time(s.start):
            continue
        retained=[w.word.strip() for w in transcript.words if s.start<=w.start<s.end
                  and mapping.time(w.end)>mapping.time(w.start)]
        # Once a span was cut, reconstruct its text from surviving raw word tokens;
        # never leave deleted false starts in the downstream planner's transcript.
        cut=abs((mapping.time(s.end)-mapping.time(s.start))-(s.end-s.start))>.02
        text=' '.join(retained) if cut and transcript.words else s.text
        if text.strip():
            segments.append(Segment(start=mapping.time(s.start),end=mapping.time(s.end),text=text))
    words = [Word(start=mapping.time(w.start), end=mapping.time(w.end), word=w.word)
             for w in transcript.words if mapping.time(w.end) > mapping.time(w.start)]
    return Transcript(segments=segments, words=words, timing_source=transcript.timing_source)


def remap_sheet(sheet, mapping):
    result = sheet.model_copy(deep=True)
    result.video_duration_sec = mapping.duration
    result.cues = [c.model_copy(update={"start_sec": mapping.time(c.start_sec), "end_sec": mapping.time(c.end_sec)})
                   for c in sheet.cues if mapping.time(c.end_sec) > mapping.time(c.start_sec)]
    seen = set()
    for cue in result.cues:
        if cue.reuse_music_from not in seen:
            cue.reuse_music_from = None
        seen.add(cue.cue_id)
    result.silence_windows = [s.model_copy(update={"start_sec": mapping.time(s.start_sec), "end_sec": mapping.time(s.end_sec)})
        for s in sheet.silence_windows if mapping.time(s.end_sec) > mapping.time(s.start_sec)]
    result.visual_events = [e.model_copy(update={"start_sec": mapping.time(e.start_sec), "end_sec": mapping.time(e.end_sec)})
        for e in sheet.visual_events if mapping.time(e.end_sec) > mapping.time(e.start_sec)]
    return SpottingSheet.model_validate(result.model_dump())


def remap_visuals(plan, mapping):
    result = []
    for event in plan.visual_events:
        if mapping.time(event.end_sec) <= mapping.time(event.start_sec):
            continue
        data = event.model_dump()
        data.update(start_sec=math.ceil(mapping.time(event.start_sec) * mapping.fps - 1e-6) / mapping.fps,
                    end_sec=min(mapping.duration, math.ceil(mapping.time(event.end_sec) * mapping.fps - 1e-6) / mapping.fps))
        for line in data.get('quote_lines',[]):
            line['start_sec']=mapping.time(line['start_sec'])
            line['word_times']=[mapping.time(t) for t in line['word_times']]
        if event.is_graphic and event.edit_type!='slogan_cards':
            needed = reading_hold(event.visible_text) + 0.3
            data["end_sec"] = max(data["end_sec"], math.ceil((data["start_sec"] + needed) * mapping.fps - 1e-6) / mapping.fps)
            if data["end_sec"] > mapping.duration + 1e-6:
                raise PipelineError("Frame-snapped graphic no longer has its required reading hold")
        result.append(VisualEditEvent.model_validate(data))
    # Revalidate graphic and zoom overlaps AFTER rounding onto the actual video clock.
    DocumentaryEditPlan(source_duration_sec=mapping.duration, visual_events=result)
    return result


def effects_and_silence(visuals):
    effects, silence = [], []
    kinds = {"sub_drop": "chapter_transition", "paper_slide_pop": "headline", "ui_data_tick": "data_reveal", "whoosh_soft": "whip"}
    for event in visuals:
        key = "edl_" + (event.event_id if len(event.event_id) <= 70 else hashlib.sha256(event.event_id.encode()).hexdigest()[:24])
        bundle={'lower_third_citation':[('ui_data_tick',0,-7)],
                'black_chapter_card':[('sub_drop',0,-2.5),('whoosh_soft',0,-6)],
                'slogan_cards':[('sub_drop',0,-2.5),('whoosh_soft',0,-6)],
                'evidence_card':[('paper_slide_pop',0,-5),('whoosh_soft',.3,-8)],
                'document_card':[('paper_slide_pop',0,-5),('whoosh_soft',.3,-8)],
                'thesis_creep':[('sub_drop',0,-10)]}.get(event.edit_type,[])
        accent_ids=[]
        for index,(name,offset,peak) in enumerate(bundle):
            fx_id=key if index==0 else key+'_layer'+str(index)
            accent_ids.append(fx_id)
            effects.append(EditEvent(event_id=fx_id, time_sec=event.start_sec+offset, kind=kinds[name],
                confidence=1, intensity=0.95, semantic_salience=0.95, required=event.required,
                documentary_card_id=event.event_id,target_peak_dbfs=peak,
                evidence="Rendered documentary visual event: " + event.edit_type))
        if event.edit_type in {"black_chapter_card","slogan_cards","thesis_creep"}:
            silence.append(SilenceWindow(start_sec=event.start_sec, end_sec=event.end_sec,
                fade_sec=.15 if event.edit_type=='thesis_creep' else .2,
                reason="Documentary quote/thesis pivot; original dialogue continues", accent_event_ids=accent_ids))
    # The music decays entirely BEFORE a chapter card, leaving its speech alone.
    return EventGraph(events=effects), silence


def remap_event_graph(graph, mapping):
    events = []
    source_duration = mapping.source_frames / mapping.fps
    for event in graph.events:
        if event.time_sec >= source_duration or (event.end_sec is not None and event.end_sec > source_duration + 0.001):
            raise PipelineError(f"Source event {event.event_id} exceeds the untrimmed video clock")
        if not mapping.retained(event.time_sec, event.end_sec or event.time_sec + 1e-4):
            if event.required:
                raise PipelineError(f"Required source event {event.event_id} lies inside a removed pause")
            continue
        data = event.model_dump()
        data["time_sec"] = mapping.time(event.time_sec)
        if event.end_sec is not None:
            data["end_sec"] = mapping.time(event.end_sec)
        events.append(EditEvent.model_validate(data))
    return EventGraph(events=events)


def prepare_documentary(config, transcript, duration, dialogue, inserts, plan_only):
    """Return an edited working config/clock. The outer run fingerprint always uses ORIGINAL inputs."""
    if config.documentary_edit_mode == "off":
        return None
    root = config.output_dir
    audio = read_audio(dialogue)[0] if dialogue else None
    retake_cuts=[]
    if config.retake_autocut and audio is not None:
        from .retake_cutter import detect_retakes
        retake_cuts,retake_decisions=detect_retakes(transcript,audio,inserts)
        save(root/'retake_analysis.json',{'clock':'source','cuts':retake_cuts,'gap_decisions':retake_decisions,
            'asr_before_visual_or_music_planning':True,'crossfade_ms':20,'retained_pause_ms':180})
    edl = plan_edl(transcript, duration, config, audio, inserts)
    if config.video_path:
        source = validate_render_source(config.video_path)
    else:
        source = {"rate": str(config.fps), "frames": round(duration * config.fps), "width": 0, "height": 0}
    mapping, trim_decisions = build_time_map(edl, transcript, audio, inserts, source, retake_cuts)
    # All secondary events use this single affine piecewise map.
    mapped = remap_transcript(transcript, mapping)
    mapped_inserts = [ArchivalInsertWindow(**{**w.model_dump(), "start_sec": mapping.time(w.start_sec),
                       "end_sec": mapping.time(w.end_sec)}) for w in inserts]
    visuals = remap_visuals(edl, mapping)
    from .evidence_fetcher import EvidenceFetcher, verify_headline_excerpt
    for event in visuals:
        if event.edit_type=='evidence_card':
            if not event.source_url:
                raise PipelineError('Evidence cards require an actual publisher source URL')
            evidence=EvidenceFetcher(root/'evidence').fetch(event.source_url)
            verify_headline_excerpt(event.headline_text,evidence['publisher_headline'])
            event.evidence_image_path=evidence['image_path']
            event.evidence_image_sha256=evidence['image_sha256']
            event.publisher=event.publisher or evidence['publisher']
    fx, silence = effects_and_silence(visuals)
    working = config.model_copy(deep=True)
    working.visual_pacing, working.pacing_detect_video, working.analyze_visuals = "off", False, False
    edited_dialogue = dialogue
    audio_qc = None
    if audio is not None:
        edited_audio, audio_qc = edit_audio(audio, mapping)
        edited_dialogue = root / "dialogue_edited.wav"
        write_audio(edited_dialogue, edited_audio)
    if config.spotting_sheet_path:
        source_sheet = SpottingSheet.model_validate_json(config.spotting_sheet_path.read_text(encoding="utf-8"))
        if abs(source_sheet.video_duration_sec - duration) > 0.05:
            raise PipelineError("Reviewed source spotting sheet must match untrimmed media")
    else:
        source_sheet = edl.music_sheet
    sheet = remap_sheet(source_sheet, mapping) if source_sheet else None
    if sheet:
        sheet.silence_windows += silence
        for cue in sheet.cues:
            if any(abs(cue.end_sec - s.start_sec) < .002 for s in silence):
                cue.exit_type, cue.exit_tail_ms, cue.exit_anchor_before_sec = "transient_stop", 150 if any(e.edit_type=='thesis_creep' and abs(e.start_sec-cue.end_sec)<.002 for e in visuals) else 200, .5
    if config.event_graph_path:
        supplied = EventGraph.model_validate_json(config.event_graph_path.read_text(encoding="utf-8"))
        fx.events.extend(remap_event_graph(supplied, mapping).events)
        fx = EventGraph.model_validate(fx.model_dump())
    # If an entire first reuse cue disappears, retain its imported source under the
    # surviving cue ID rather than accidentally switching to an unused manifest entry.
    if source_sheet and sheet:
        from .providers import read_manifest
        lookup = {c.cue_id: c for c in source_sheet.cues}
        for field in ("local_assets", "google_flow_assets", "elevenlabs_assets"):
            path = getattr(config, field)
            if not path:
                continue
            assets = read_manifest(path)
            for cue in sheet.cues:
                original = lookup[cue.cue_id]
                source_id = original.cue_id
                while lookup[source_id].reuse_music_from:
                    source_id = lookup[source_id].reuse_music_from
                if not cue.reuse_music_from and original.reuse_music_from and source_id in assets["music"]:
                    assets["music"][cue.cue_id] = assets["music"][source_id]
                    for key in ("downbeats", "loops"):
                        if source_id in assets[key]:
                            assets[key][cue.cue_id] = assets[key][source_id]
            data = {key: {k: str(v) for k, v in value.items()} if key in {"music", "sfx"} else value for key, value in assets.items()}
            derived = root / f"edited_{field}.json"
            save(derived, data)
            setattr(working, field, derived)
    if config.locked_music_path:
        from .music_continuity import sha256
        if sha256(config.locked_music_path) != config.locked_music_sha256:
            raise PipelineError("Locked music changed before documentary retiming")
        music, rate = read_audio(config.locked_music_path)
        if rate != SR or len(music) != round(duration * SR):
            raise PipelineError("Locked source music does not match the untrimmed source clock")
        music, _ = edit_audio(music, mapping, bridge=False)
        if sheet is None:
            raise PipelineError("A documentary music lock requires its source spotting sheet")
        from .archival import protect_archival
        from .dsp import silence_envelope
        muted_sheet = protect_archival(sheet, mapped_inserts)
        music *= silence_envelope(len(music), 0, muted_sheet)[:, None]
        # Recorded-music early reflections decay before the exact thesis silence boundary.
        # This is convolution/delay of the actual bed, never a synthesized drone/tone.
        for event in visuals:
            if event.edit_type!='thesis_creep':
                continue
            end=round(event.start_sec*SR)
            length=round(.15*SR)
            start=max(0,end-length)
            dry=music[start:end].copy()
            tail=np.zeros_like(dry)
            for delay,gain in ((.013,.22),(.023,.16),(.037,.11),(.053,.07)):
                lag=round(delay*SR)
                if lag<len(tail):
                    tail[lag:]+=dry[:-lag]*gain
            decay=np.linspace(1,0,len(tail),dtype=np.float32)**2
            music[start:end]+=tail*decay[:,None]
            music[end:round(event.end_sec*SR)]=0
        working.locked_music_path = root / "locked_music_edited.wav"
        write_audio(working.locked_music_path, music)
        working.locked_music_sha256 = sha256(working.locked_music_path)
    report = {"source_duration_sec": duration, "output_duration_sec": mapping.duration,
        "removed_sec": duration - mapping.duration, "trim_decisions": trim_decisions, "audio": audio_qc,
        "original_music_sha256": config.locked_music_sha256, "edited_music_sha256": working.locked_music_sha256,
        "visual_render": {"status": "planned"}, "warnings": edl.warnings}
    if not plan_only and config.mode != "script_only":
        if config.documentary_edit_mode != "render":
            raise PipelineError("Use --plan-only for documentary plan mode, or choose render/off for asset/video output")
        if not config.video_path:
            raise PipelineError("Documentary visual rendering needs recorded video")
        from .video_editor_engine import render_documentary_video_track
        working.video_path = root / "documentary_video.mp4"
        report["visual_render"] = render_documentary_video_track(config.video_path, working.video_path, visuals,
                                                               mapping, source, config.documentary_font_path,
                                                               word_timestamps=[w.model_dump() for w in mapped.words],
                                                               cache_root=root / "html_overlays",
                                                               studio_grade=config.studio_commentary_grade)
    save(root / "documentary_render_qc.json", report)
    save(root / "timeline_map.json", mapping.model_dump())
    save(root / "source_transcript.json", transcript.model_dump())
    save(root / "transcript.json", mapped.model_dump())
    if mapped.words:
        save(root / "word_timestamps.json", [w.model_dump() for w in mapped.words])
    save(root / "archival_windows_edited.json", {"clock": "edited", "windows": [w.model_dump() for w in mapped_inserts]})
    save(root / "documentary_visual_plan.json", {"clock": "edited", "events": [e.model_dump() for e in visuals]})
    save(root / "documentary_map_fingerprint.json", {"sha256": hashlib.sha256(mapping.model_dump_json().encode()).hexdigest()})
    return {"config": working, "transcript": mapped, "duration": mapping.duration, "dialogue": edited_dialogue,
            "inserts": mapped_inserts, "sheet": sheet, "graph": fx, "silence": silence, "report": report}
