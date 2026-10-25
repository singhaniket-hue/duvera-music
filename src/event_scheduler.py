"""Conservative evidence -> editorial instructions, with inspectable rejection reasons."""
from __future__ import annotations

from .models import RunConfig, SpottingSheet, Transcript
from .pacing_models import EventGraph, PacingPlan, SFXInstruction, VisualInstruction

PRESETS = {
    "headline": ("paper_slide", 0.22, -21, -7, 8, "Soft paper slide and muted low-mid pop, no voice, no music"),
    "data_reveal": ("ui_tick", 0.06, -21, -7, 6, "Restrained mechanical evidence counter tick, no voice, no music"),
    "chapter_transition": ("sub_drop", 0.55, -18, -7, 10, "Restrained dark 45–80Hz documentary sub, no voice, no music"),
    "riser": ("riser", 0.45, -20, -7, 0, "Soft non-tonal documentary transition riser, no voice, no music"),
    "black_card": ("sub_drop", 0.55, -18, -7, 10,
                   "Restrained warm 35–80 Hz sub thud, no midrange click, no voice, no music"),
    "text_enter": ("text_pop", 0.08, -21, -7, 6,
                   "Tiny muted low-mid wooden text pop, soft attack, no harsh treble, no voice, no music"),
    "whip": ("whoosh", 0.3, -20, -7, 8,
             "Short restrained air whoosh, soft high frequencies, no impact, no voice, no music"),
    "zoom": ("whoosh", 0.22, -20, -7, 8,
             "Tiny soft ascending air whoosh, no impact, no voice, no music"),
    "impact": ("impact", 0.3, -20, -7, 8, "Restrained soft low impact, no voice, no music"),
    "ui_tick": ("ui_tick", 0.06, -21, -7, 6, "Tiny soft tactile interface tick, no voice, no music"),
    "shutter": ("shutter", 0.12, -21, -7, 0, "Soft restrained camera shutter, no voice, no music"),
}


def protected(time: float, sheet: SpottingSheet) -> bool:
    return any(s.start_sec <= time < s.end_sec for s in sheet.silence_windows)


def archival_protected(time: float, sheet: SpottingSheet) -> bool:
    return any(s.kind == "archival" and s.start_sec <= time < s.end_sec for s in sheet.silence_windows)


def accent_authorized(time, event_id, sheet, config):
    return not archival_protected(time, sheet) and (config.sfx_silence_policy == "accents" or any(
        s.start_sec <= time < s.end_sec and event_id in s.accent_event_ids for s in sheet.silence_windows))


def schedule(graph: EventGraph, sheet: SpottingSheet, transcript: Transcript, config: RunConfig) -> PacingPlan:
    result = PacingPlan()
    duration = sheet.video_duration_sec
    last_accent, last_major, last_visual = -100.0, -100.0, 0.0
    location = None
    events = sorted(graph.events, key=lambda e: (e.time_sec, -e.confidence, e.event_id))
    for index, event in enumerate(events):
        if event.time_sec >= duration or (event.end_sec is not None and event.end_sec > duration + 0.001):
            raise ValueError(f"Event {event.event_id} exceeds source duration")
        dialogue_active = any(s.start <= event.time_sec < s.end for s in transcript.segments)
        structural = event.kind in {"black_card", "text_enter", "whip", "zoom", "ui_tick", "shutter",
                                    "headline", "data_reveal", "chapter_transition", "riser"}
        score = (0.30 * event.semantic_salience + 0.25 * event.intensity + 0.20 * event.confidence
                 + 0.15 * structural + 0.10 * (not dialogue_active))
        decision = {"event_id": event.event_id, "time_sec": event.time_sec,
                    "score": round(score, 4), "dialogue_active": dialogue_active, "accepted": False,
                    "required": event.required}
        result.decisions.append(decision)
        if event.kind == "punch_in":
            if config.visual_pacing == "off":
                decision["reason"] = "visual pacing disabled"
            elif event.confidence < 0.8 or event.eye_anchor is None or event.end_sec is None:
                decision["reason"] = "needs stable eye anchor, confidence >=0.8 and reviewed end"
            elif event.time_sec < last_visual or any(event.time_sec < s.end_sec and event.end_sec > s.start_sec
                                                      for s in sheet.silence_windows):
                decision["reason"] = "overlapping visual instruction or protected silence"
            else:
                result.visuals.append(VisualInstruction(event_id=event.event_id, start_sec=event.time_sec,
                    end_sec=event.end_sec, eye_anchor=event.eye_anchor, zoom_to=event.zoom_factor))
                last_visual = event.end_sec
                decision.update(accepted=True, reason="anchored visual instruction")
            continue
        if config.sfx_mode == "none":
            decision["reason"] = "SFX extension disabled"
            continue
        if event.confidence < 0.7:
            decision["reason"] = "low confidence"
            continue
        if config.preset == "social_commentary" and event.kind in {"location_change", "foley"}:
            decision["reason"] = "documentary palette excludes invented environmental ambience and object Foley"
            continue
        if event.kind == "location_change":
            if config.sfx_mode != "full_3_tier":
                decision["reason"] = "ambience requires full_3_tier"
                continue
            if location == event.location:
                decision["reason"] = "same acoustic location held across angles"
                continue
            following = next((x for x in events[index + 1:] if x.kind == "location_change"
                              and x.confidence >= 0.7 and x.location != event.location), None)
            end = min(event.end_sec or duration, following.time_sec + 0.55 if following else duration)
            start = max(0, event.time_sec - 0.45)
            if result.effects and location:
                for previous in reversed(result.effects):
                    if previous.tier == "ambience":
                        previous.end_sec = min(previous.end_sec, event.time_sec + 0.55)
                        break
            result.effects.append(SFXInstruction(asset_id="fx_" + event.event_id, event_id=event.event_id,
                visual_time_sec=event.time_sec, end_sec=end, tier="ambience", category="ambience",
                prompt=event.prompt + ", continuous room tone, no foreground events, no intelligible voices, no music",
                duration_sec=min(30, end - start), generation_sec=min(30, max(0.5, end - start)),
                target_lufs=-32, peak_ceiling_db=-10, j_lead_sec=0.45, crossfade_sec=1.0))
            location = event.location
            decision.update(accepted=True, reason="evidenced acoustic location; 450ms ambience J lead")
            continue
        if event.kind == "foley":
            if config.sfx_mode != "full_3_tier" or protected(event.time_sec, sheet):
                decision["reason"] = "Foley disabled or protected silence"
                continue
            length = min(30, (event.end_sec or (event.time_sec + 0.4)) - event.time_sec)
            result.effects.append(SFXInstruction(asset_id="fx_" + event.event_id, event_id=event.event_id,
                visual_time_sec=event.time_sec, tier="foley", category="foley", prompt=event.prompt,
                duration_sec=length, generation_sec=max(0.5, length), target_lufs=-21, peak_ceiling_db=-6))
            decision.update(accepted=True, reason="evidenced object sound")
            continue
        if event.kind not in PRESETS:
            decision["reason"] = "ordinary cut has no independent accent evidence"
            continue
        if archival_protected(event.time_sec, sheet):
            decision["reason"] = "archival audio owns the mix; no added editorial accent"
            continue
        if event.time_sec + 0.025 >= duration:
            decision["reason"] = "insufficient timeline for acoustic landmark"
            continue
        if protected(event.time_sec, sheet) and not accent_authorized(event.time_sec, event.event_id, sheet, config):
            decision["reason"] = "protected punchline silence"
            continue
        major = event.kind in {"black_card", "impact"}
        if not event.documentary_card_id and (event.time_sec - last_accent < config.accent_cooldown_sec or
                (major and event.time_sec - last_major < config.major_cooldown_sec)):
            decision["reason"] = "refractory cooldown"
            continue
        if score < config.accent_salience_threshold:
            decision["reason"] = "insufficient salience"
            continue
        category, length, target, ceiling, offset, prompt = PRESETS[event.kind]
        if category == "whoosh":
            length = 0.10 + 0.35 * event.intensity ** 0.7
        result.effects.append(SFXInstruction(asset_id="fx_" + event.event_id, event_id=event.event_id,
            visual_time_sec=event.time_sec, tier="accent", category=category, prompt=event.prompt or prompt,
            duration_sec=length, generation_sec=max(0.5, length), target_lufs=target,
            peak_ceiling_db=ceiling, av_offset_ms=offset))
        last_accent = event.time_sec
        if major:
            last_major = event.time_sec
        decision.update(accepted=True, reason="salience and cooldown passed")
    # Review candidates only: never delete breath/silence or retime visible speech automatically.
    for left, right in zip(transcript.segments, transcript.segments[1:]):
        gap = right.start - left.end
        if gap > 0.45:
            result.pause_review.append({"start_sec": left.end, "end_sec": right.start,
                "action": "hold" if protected(left.end, sheet) else "review_only",
                "reason": "ASR/segment gap is not proof of disposable silence; breath and visual review required"})
    if config.sfx_mode == "full_3_tier" and not any(x.tier == "ambience" for x in result.effects):
        result.warnings.append("No evidenced location events: ambience was not invented. Supply an event graph.")
    lookup = {e.event_id: e for e in graph.events}
    for effect in result.effects:
        effect.required = lookup[effect.event_id].required
        if config.documentary_edit_mode != "off" and effect.tier == "accent":
            effect.av_offset_ms = 10 if effect.category == "sub_drop" else 8
            effect.generation_sec = 1.0
        if config.sfx_mix_profile == "audible":
            if effect.tier == "ambience":
                effect.level_mode, effect.target_rms_dbfs = "active_rms", -28
            else:
                effect.level_mode = "sample_peak"
                effect.target_peak_dbfs = -3.5 if effect.category == "sub_drop" else -7
                effect.peak_ceiling_db = -3.5 if effect.category == "sub_drop" else -5
        event=lookup[effect.event_id]
        if event.documentary_card_id and event.target_peak_dbfs is not None:
            effect.level_mode='sample_peak'
            effect.target_peak_dbfs=event.target_peak_dbfs
            effect.peak_ceiling_db=event.target_peak_dbfs+.5
            effect.duration_sec={'sub_drop':.8,'paper_slide':.4,'ui_tick':.16,'whoosh':.45}.get(effect.category,.3)
    for decision in result.decisions:
        if decision["required"] and not decision["accepted"]:
            result.blocking_issues.append(f"Required event {decision['event_id']}: {decision['reason']}")
    if config.sfx_mode != "none" and not result.effects:
        result.blocking_issues.append("SFX enabled but no effects accepted; review mode, silence policy and event evidence")
    return result
