"""Shared analysis, reproducible planning, pre-generation gates and isolated provider branches."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

from .analyzer import ingest, prosodic_pulse
from .director import plan
from .dsp import build_stems, render
from .dsp_mixer import render_effects
from .event_scheduler import schedule
from .exporter import export_script, export_timeline
from .media import PipelineError, SR, loudness, require_ffmpeg
from .models import RunConfig, SpottingSheet
from .pacing_models import EventGraph, PacingPlan
from .providers import AssetStore, generation_jobs, validate_manifest_for_sheet
from .providers.elevenlabs import ElevenLabsProvider
from .providers.google_flow import GoogleFlowProvider, load_browser_config
from .visuals import analyze_video
from .vocals import VocalGuard
from .visual_pacing import validate_render_source
from .music_continuity import build_locked_stems, music_gap_qc, sha256, validate_locked_music


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _fingerprint(config: RunConfig) -> str:
    data = config.model_dump(mode="json")
    for field in ("allow_paid_generation", "max_generated_seconds", "max_requests"):
        data.pop(field)
    files = {}
    for name, value in config:
        if name.endswith("_path") or name.endswith("_assets") or name == "google_flow_browser_config":
            if value is not None:
                if not value.is_file():
                    raise PipelineError(f"Input file not found: {value}")
                stat = value.stat()
                files[name] = [str(value.resolve()), stat.st_size, stat.st_mtime_ns]
    return hashlib.sha256(json.dumps(["sfx-audibility-v2", data, files], sort_keys=True).encode()).hexdigest()


def _branches(config: RunConfig):
    if config.provider == "both_compare":
        return [("elevenlabs", config.elevenlabs_assets), ("google_flow", config.google_flow_assets)]
    manifest = {"local": config.local_assets, "elevenlabs": config.elevenlabs_assets,
                "google_flow": config.google_flow_assets}.get(config.provider)
    return [(config.provider, manifest)]


def _gate(config: RunConfig, sheet: SpottingSheet, pacing=None):
    """Validate BOTH branches and the entire generation budget before any paid request."""
    planned = {}
    for name, manifest in _branches(config):
        if manifest:
            validate_manifest_for_sheet(manifest, sheet, pacing, bool(config.locked_music_path))
            planned[name] = {"mode": "imported", "requests": 0, "seconds": 0}
            continue
        jobs = list(generation_jobs(sheet, pacing, bool(config.locked_music_path)))
        if name == "google_flow" and pacing and pacing.effects:
            raise PipelineError("Pacing SFX on the Google branch require imported assets; the music browser UI "
                                "is not a verified three-tier SFX generator. Supply --google-flow-assets.")
        seconds = sum(j[2] for j in jobs)
        if len(jobs) > config.max_requests or seconds > config.max_generated_seconds:
            raise PipelineError(f"{name} needs {len(jobs)} requests / {seconds:.1f}s, exceeding configured budget")
        if any(not 0.5 <= duration <= (30 if kind == "sfx" else 600) for _, kind, duration, _ in jobs):
            raise PipelineError("Invalid cue duration for generation")
        if not config.allow_paid_generation:
            raise PipelineError(f"{name} generation requires --allow-paid-generation. "
                                "Review generation_plan.json first; supplied assets need no spend authorization.")
        if name == "elevenlabs":
            if not os.getenv("ELEVENLABS_API_KEY", "").strip():
                raise PipelineError("ElevenLabs key unavailable; no providers were called")
        elif name == "google_flow":
            if not config.google_flow_browser_config:
                raise PipelineError("Google Flow requires --google-flow-assets or a verified browser configuration")
            load_browser_config(config.google_flow_browser_config)
        elif name == "local":
            raise PipelineError("Local assets manifest is required")
        planned[name] = {"mode": "generated", "requests": len(jobs), "seconds": seconds}
    return planned


def execute(config: RunConfig, plan_only: bool = False) -> dict:
    fingerprint = _fingerprint(config)
    root = config.output_dir
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two processes from spending on the same pending cue.
    lock = root / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PipelineError("Output is locked. Check for a running process before removing .run.lock") from exc
    os.close(fd)
    try:
        old = root / "run_fingerprint.txt"
        if old.exists() and old.read_text().strip() != fingerprint:
            raise PipelineError("Inputs/config changed; use a new output directory to avoid mixing stale artifacts")
        old.write_text(fingerprint)
        return _execute(config, plan_only)
    finally:
        lock.unlink(missing_ok=True)


def _execute(config: RunConfig, plan_only: bool) -> dict:
    root = config.output_dir
    if config.video_path or config.mode != "script_only":
        require_ffmpeg()
    transcript, duration, dialogue = ingest(config)
    visuals = analyze_video(config.video_path, duration, config) if config.video_path else []
    save_json(root / "visual_analysis.json", {
        "events": [v.model_dump() for v in visuals],
        "enabled": config.analyze_visuals,
        "resolution_sec": 1 / config.visual_sample_fps,
        "note": "Luminance and sparse highlights suggest cards; no OCR or semantic text detection.",
    })
    pulse, dialogue_lufs = None, None
    if dialogue:
        dialogue_lufs = loudness(dialogue)["input_i"]
        with wave.open(str(dialogue), "rb") as w:
            sample = np.frombuffer(w.readframes(60 * SR), "<i2").astype(np.float32).reshape(-1, 2) / 32768
        pulse = prosodic_pulse(sample)
    if config.spotting_sheet_path:
        sheet = SpottingSheet.model_validate_json(config.spotting_sheet_path.read_text(encoding="utf-8"))
        if abs(sheet.video_duration_sec - duration) > 0.001:
            raise PipelineError("Edited spotting sheet duration must match media/transcript")
        if sheet.timing_source != transcript.timing_source:
            raise PipelineError("Spotting sheet timing_source differs from transcript")
        sheet.dialogue_integrated_lufs = dialogue_lufs
    elif (root / "spotting_sheet.json").exists():
        sheet = SpottingSheet.model_validate_json((root / "spotting_sheet.json").read_text(encoding="utf-8"))
    else:
        sheet = plan(transcript, duration, config, dialogue_lufs, pulse, visuals)
    if abs(sheet.video_duration_sec - duration) > 0.001 or sheet.timing_source != transcript.timing_source:
        raise PipelineError("Cached spotting sheet no longer matches media timing; use a new output directory")
    (root / "spotting_sheet.json").write_text(sheet.model_dump_json(indent=2), encoding="utf-8")
    sheet_hash = hashlib.sha256(sheet.model_dump_json().encode()).hexdigest()
    pacing_sheet_hash = root / "pacing_sheet_hash.txt"
    if (root / "pacing_plan.json").exists():
        if not pacing_sheet_hash.exists() or pacing_sheet_hash.read_text().strip() != sheet_hash:
            raise PipelineError("Spotting sheet changed after pacing was planned; use a fresh output directory")
        graph = EventGraph.model_validate_json((root / "event_graph.json").read_text(encoding="utf-8"))
        pacing = PacingPlan.model_validate_json((root / "pacing_plan.json").read_text(encoding="utf-8"))
    else:
        graph = EventGraph.model_validate_json(config.event_graph_path.read_text(encoding="utf-8")) if config.event_graph_path else EventGraph()
        enabled = config.sfx_mode != "none" or config.visual_pacing != "off"
        if enabled and config.pacing_detect_video and config.video_path:
            from .cv_detectors import detect_video
            from .narrative import narrative_events
            candidates = [e for e in narrative_events(transcript) if e["kind"] in {"contrast", "emphasis"}] if config.visual_pacing != "off" else []
            detected, analysis = detect_video(config.video_path, duration, candidates)
            additions = [e for e in detected.events if not any(e.kind == supplied.kind and
                         abs(e.time_sec - supplied.time_sec) < 0.35 for supplied in graph.events)]
            graph = EventGraph(events=graph.events + additions)
            save_json(root / "pacing_analysis.json", analysis)
        pacing = schedule(graph, sheet, transcript, config)
    visual_source = validate_render_source(config.video_path) if config.visual_pacing == "render" else None
    save_json(root / "event_graph.json", graph.model_dump(mode="json"))
    save_json(root / "pacing_plan.json", pacing.model_dump(mode="json"))
    pacing_sheet_hash.write_text(sheet_hash + "\n")
    export_script(sheet, transcript, config, root)
    jobs = [{"cue_id": c.cue_id, "kind": kind, "duration_seconds": seconds, "prompt": prompt,
             "loop": getattr(c, "loop", False)}
            for c, kind, seconds, prompt in generation_jobs(sheet, pacing, bool(config.locked_music_path))]
    save_json(root / "generation_plan.json", {
        "branches": [name for name, _ in _branches(config)] if config.mode != "script_only" else [],
        "jobs_per_generated_branch": jobs,
        "max_requests_per_provider": config.max_requests,
        "max_generated_seconds_per_provider": config.max_generated_seconds,
        "sfx_mode": config.sfx_mode, "sfx_silence_policy": config.sfx_silence_policy,
        "sfx_mix_profile": config.sfx_mix_profile, "accepted_sfx_count": len(pacing.effects),
        "blocking_issues": pacing.blocking_issues,
        "locked_music_sha256": config.locked_music_sha256,
        "note": "Duration/request caps are not monetary price estimates. Imports do not make provider calls.",
    })
    if config.mode == "script_only" or plan_only:
        return {"status": "planned", "output_dir": str(root.resolve()), "timing_source": sheet.timing_source,
                "sfx_count": len(pacing.effects), "blocking_issues": pacing.blocking_issues}
    if pacing.blocking_issues:
        raise PipelineError("; ".join(pacing.blocking_issues))
    if config.locked_music_path:
        validate_locked_music(config, duration, sheet)
    planned = _gate(config, sheet, pacing)
    guard = VocalGuard(config, root / "vocal_checks")
    # Validate checker/model/device before any provider generates credit-consuming audio.
    if config.locked_music_path or any(kind == "music" for _, kind, _, _ in generation_jobs(sheet)):
        guard.prepare()
    if config.locked_music_path:
        approved = guard.inspect("locked/music", config.locked_music_path)
        if sha256(approved) != config.locked_music_sha256:
            raise PipelineError("Vocal stripping would alter locked music. Resolve/review the bed before locking it.")
    providers = {}
    try:
        # Open/check Google first, before spending on ElevenLabs in comparison mode.
        if any(name == "google_flow" and not manifest for name, manifest in _branches(config)):
            providers["google_flow"] = GoogleFlowProvider(config.google_flow_browser_config, config.chrome_profile_dir)
        if any(name == "elevenlabs" and not manifest for name, manifest in _branches(config)):
            providers["elevenlabs"] = ElevenLabsProvider()
        results = {}
        for name, manifest in _branches(config):
            branch = root / name
            branch.mkdir(exist_ok=True)
            store = AssetStore(branch / "stems", config, manifest, providers.get(name))
            store.safe_music = {}
            for cue, kind, seconds, prompt in generation_jobs(sheet, pacing, bool(config.locked_music_path)):
                asset = store.get(cue.cue_id, prompt, seconds, kind, loop=getattr(cue, "loop", False))
                if kind == "music":
                    store.safe_music[cue.cue_id] = guard.inspect(f"{name}/{cue.cue_id}", asset)
            music, sfx, records = (build_locked_stems(sheet, config, branch) if config.locked_music_path else
                                   build_stems(sheet, store, dialogue, branch, config))
            continuity = music_gap_qc(music, sheet, config.max_music_gap_sec)
            sfx_qc = render_effects(pacing, sheet, store, dialogue, branch, config) if pacing.effects else None
            if sfx_qc:
                sfx = branch / "sfx_stem.wav"
                for item in sfx_qc["effects"]:
                    print(json.dumps({"sfx_event": item["event_id"], "tier": item.get("tier"),
                        "start_sec": item.get("start_sample", 0) / SR,
                        "peak_sec": item["actual_peak_sample"] / SR if item.get("actual_peak_sample") is not None else None,
                        "routed_sample_peak_dbfs": item.get("routed_sample_peak_dbfs"),
                        "active_rms_dbfs": item.get("active_rms_dbfs"), "status": item["status"],
                        "passed": item["passed"]}, ensure_ascii=False), file=sys.stderr)
            export_script(sheet, transcript, config, branch)
            export_timeline(sheet, config, branch, records, dialogue, pacing)
            qc = {"provider": name, "source": planned[name], "cues": records, "mastering": None,
                  "vocal_policy": config.vocal_policy, "sfx_pacing": sfx_qc, "music_continuity": continuity,
                  "locked_music_sha256": config.locked_music_sha256,
                  "locked_music_vocal_check": guard.reports.get("locked/music"),
                  "vocal_checks": {key.split("/", 1)[1]: value for key, value in guard.reports.items()
                                   if key.startswith(name + "/")}}
            if config.mode == "final_video" and continuity["passed"] and (sfx_qc is None or sfx_qc["passed"]):
                qc["mastering"] = render(dialogue, music, sfx, branch, config, pacing, visual_source)
            elif config.mode == "final_video":
                qc["mastering"] = {"passed": False, "encoded_measurement": {},
                                   "note": "Premux SFX/continuity QC failed; no final video was muxed"}
            save_json(branch / "qc_report.json", qc)
            results[name] = qc
        if config.provider == "both_compare":
            comparison_report(results, root)
        qc_passed = all((q["mastering"] is None or q["mastering"]["passed"]) and
                        q["music_continuity"]["passed"] and
                        (q["sfx_pacing"] is None or q["sfx_pacing"]["passed"]) for q in results.values())
        summary = {"status": "complete" if qc_passed else "qc_failed",
                   "output_dir": str(root.resolve()), "providers": list(results),
                   "timing_source": sheet.timing_source,
                   "sfx_mode": config.sfx_mode, "sfx_count": len(pacing.effects),
                   "sfx_silence_policy": config.sfx_silence_policy, "sfx_mix_profile": config.sfx_mix_profile,
                   "vocal_check": ("not_required_no_music" if not guard.reports else
                                   "disabled_unchecked" if config.vocal_policy == "off" else "model_proxy_checked")}
        save_json(root / "run_summary.json", summary)
        return summary
    finally:
        if "google_flow" in providers:
            providers["google_flow"].close()


def comparison_report(results: dict, root: Path):
    lines = ["# Provider comparison", "",
             "Shared transcript, spotting sheet, gaps, silences and mastering preset. "
             "Music content differs; these measurements do not determine artistic quality.", "",
             "| Provider | Encoded LUFS | Encoded dBTP | Mean speech-band music/voice dB | Verified downbeats | QC |",
             "|---|---:|---:|---:|---|---|"]
    for name, report in results.items():
        master = report["mastering"]
        encoded = master["encoded_measurement"] if master else {}
        overlaps = [c["speech_band_music_to_voice_db"] for c in report["cues"]
                    if c["speech_band_music_to_voice_db"] is not None]
        overlap = round(sum(overlaps) / len(overlaps), 2) if overlaps else "N/A"
        verified = sum(bool(c["analysis"]["downbeat_verified"]) for c in report["cues"])
        lines.append(f"| {name} | {encoded.get('input_i', 'N/A')} | {encoded.get('input_tp', 'N/A')} | "
                     f"{overlap} | {verified}/{len(report['cues'])}, otherwise onset estimates | "
                     f"{master['passed'] if master else 'not mastered'} |")
    lines += ["", "Speech-band ratio is a band-energy proxy, not a validated intelligibility score. "
              "Downbeat alignment accuracy is unavailable without annotated ground truth. "
              "Listen to both outputs at equal playback volume before selecting."]
    (root / "ab_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
