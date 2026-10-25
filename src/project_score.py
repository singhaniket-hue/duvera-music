"""Master an assembled real documentary with checked instrumental chapter sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .archival import ArchivalInsertWindow, protect_archival
from .dialogue_polish import polish_dialogue
from .documentary_audio import build_documentary_sfx_stem
from .dsp import build_stems, render
from .event_scheduler import schedule
from .media import PipelineError, SR, ffmpeg, read_audio
from .models import MusicCue, RunConfig, Segment, SilenceWindow, SpottingSheet, Transcript, Word
from .music_continuity import music_gap_qc
from .pacing_models import EventGraph
from .production_project import save, sha, video_info
from .providers import AssetStore


def assemble_transcript(root, assembly):
    words = []
    segments = []
    for piece in assembly["segments"]:
        if piece["kind"] != "camera":
            continue
        transcript = Transcript.model_validate_json(
            (root / "parts" / piece["part_id"] / "transcript.json").read_text(encoding="utf-8")
        )
        a, b = piece["source_start_frame"] / 60, piece["source_end_frame"] / 60
        offset = piece["output_start_frame"] / 60 - a
        for word in transcript.words:
            if a <= word.start < b and min(b, word.end) > word.start:
                words.append(Word(start=word.start + offset, end=min(b, word.end) + offset, word=word.word))
        for segment in transcript.segments:
            start, end = max(a, segment.start), min(b, segment.end)
            if end > start:
                segments.append(Segment(start=start + offset, end=end + offset, text=segment.text))
    return Transcript(timing_source="asr", words=words, segments=segments)


def score_project(project_path, manifest_path):
    project = json.loads(Path(project_path).read_text(encoding="utf-8"))
    root = Path(project["output_dir"])
    assembly = json.loads((root / "assembly_plan.json").read_text(encoding="utf-8"))
    duration = assembly["frames"] / 60
    target = root / "mix"
    target.mkdir(exist_ok=True)
    archives = [
        ArchivalInsertWindow(**a) for a in json.loads((root / "archival_windows.json").read_text())["windows"]
    ]
    dialogue, polish = polish_dialogue(root / "dialogue_assembled.wav", archives, target)
    if not polish.get("passed", True):
        raise PipelineError("Full-project narration polish failed")
    transcript = assemble_transcript(root, assembly)
    save(root / "full_transcript.json", transcript.model_dump())
    windows = [
        SilenceWindow(**w) for w in json.loads((root / "silence_windows.json").read_text(encoding="utf-8"))
    ]
    closing = next(
        s["output_start_frame"] / 60
        for s in assembly["segments"]
        if s["kind"] == "camera" and s["part_id"] == project["parts"][-1]["id"]
    )
    windows.append(
        SilenceWindow(
            start_sec=closing,
            end_sec=duration,
            fade_sec=0.2,
            reason="Closing question delivered with original voice alone",
        )
    )
    manifest = json.loads(Path(manifest_path).read_text())
    vocal_path = manifest.get("vocal_qc_path")
    if not vocal_path:
        raise PipelineError("A full-project music manifest needs its VocalGuard receipt")
    vocal_report = json.loads(Path(vocal_path).read_text(encoding="utf-8"))
    checked = {
        item.get("selected_sha256")
        for item in vocal_report["assets"].values()
        if item.get("status") in {"below_proxy_threshold", "stripped_and_rechecked"}
    }
    if vocal_report["policy"] == "off" or any(
        sha(path) not in checked for path in manifest["music"].values()
    ):
        raise PipelineError("Full-project music changed or lacks a successful vocal check")
    cues = []
    moods = [
        (
            "Opening / the online trend",
            96,
            ["curious", "restrained", "warm"],
            ["felt piano", "muted plucks", "soft low pulse"],
        ),
        (
            "Expectations / commentary",
            92,
            ["analytical", "thoughtful", "dry"],
            ["muted acoustic guitar", "felt piano", "brushed percussion"],
        ),
        (
            "Safety / lived experience",
            76,
            ["serious", "empathetic", "grounded"],
            ["low felt piano", "muted bass", "soft brushed texture"],
        ),
        (
            "Accountability / reflection",
            84,
            ["reflective", "calm", "restrained resolution"],
            ["nylon guitar harmonics", "felt piano", "brushed shaker"],
        ),
    ]
    for index, (title, bpm, mood, instruments) in enumerate(moods):
        start, end = duration * index / 4, duration * (index + 1) / 4
        if not 90 <= end - start <= 150:
            raise PipelineError(
                "This four-theme score requires 6–10 minutes; provide a different chapter plan"
            )
        cues.append(
            MusicCue(
                cue_id=f"cue_{index + 1:03}",
                chapter_id=f"chapter_{index + 1:03}",
                chapter_title=title,
                start_sec=start,
                end_sec=end,
                narrative_purpose="Sparse instrumental chapter bed for continuous real commentary: " + title,
                speech_syllables_per_sec=4,
                target_bpm=bpm,
                mood_descriptors=mood,
                allowed_instruments=instruments,
                depth_profile="background",
                target_lu_gap=18,
                generation_seconds=45,
            )
        )
    sheet = SpottingSheet(
        video_duration_sec=duration,
        timing_source="asr",
        planner="reviewed_full_project",
        cues=cues,
        silence_windows=windows,
    )
    sheet = protect_archival(sheet, archives)
    config = RunConfig(
        preset="social_commentary",
        video_path=root / "full_picture.mp4",
        transcript_path=root / "full_transcript.json",
        output_dir=target,
        mode="final_video",
        provider="elevenlabs",
        documentary_edit_mode="render",
        language="hinglish",
        allow_paid_generation=False,
        max_requests=3,
        max_generated_seconds=135,
        sfx_mode="accents_only",
        sfx_mix_profile="audible",
        dialogue_polish=True,
        vocal_policy="reject",
        target_lufs=-15,
        true_peak_db=-1.5,
    )
    graph = EventGraph.model_validate_json((root / "event_graph.json").read_text(encoding="utf-8"))
    pacing = schedule(graph, sheet, transcript, config)
    if pacing.blocking_issues:
        raise PipelineError("; ".join(pacing.blocking_issues))
    save(root / "full_spotting_sheet.json", sheet.model_dump())
    save(root / "full_pacing_plan.json", pacing.model_dump())
    store = AssetStore(target / "stems", config, Path(manifest_path), None)
    store.safe_music = {key: Path(value) for key, value in manifest["music"].items()}
    store.acoustic = {effect.asset_id: effect for effect in pacing.effects}
    for effect in pacing.effects:
        store.get(effect.asset_id, effect.prompt, effect.generation_sec, "sfx")
    music, sfx, records = build_stems(sheet, store, dialogue, target, config)
    continuity = music_gap_qc(music, sheet, config.max_music_gap_sec)
    if not continuity["passed"]:
        raise PipelineError("Unexpected full-score music gap")
    effects = build_documentary_sfx_stem(pacing, sheet, store, dialogue, target, config)
    if not effects["passed"]:
        raise PipelineError("Full-project effects stem failed")
    source = {"rate": "60", "frames": assembly["frames"], "width": 1920, "height": 1080}
    mix = render(dialogue, music, target / "sfx_stem.wav", target, config, pacing, source)
    if not mix["passed"]:
        raise PipelineError("Full-project encoded mastering failed")
    video = target / ("final_video_" + target.name + ".mp4")
    ffmpeg("-v", "error", "-i", video, "-f", "null", "-")
    source_sound, _ = read_audio(root / "dialogue_assembled.wav")
    polished, _ = read_audio(dialogue)
    bed, _ = read_audio(music)
    fx, _ = read_audio(target / "sfx_stem.wav")
    protected = []
    for archive in archives:
        a, b = round(archive.start_sec * SR), round(archive.end_sec * SR)
        protected.append(
            {
                "start_sec": archive.start_sec,
                "end_sec": archive.end_sec,
                "source": archive.source,
                "original_samples_preserved_before_master": np.array_equal(source_sound[a:b], polished[a:b]),
                "added_music_zero": float(np.max(abs(bed[a:b]), initial=0)) == 0,
                "added_effects_zero": float(np.max(abs(fx[a:b]), initial=0)) == 0,
            }
        )
    decoded = next(
        s
        for s in assembly["segments"]
        if s["kind"] == "camera" and s["part_id"] == project["parts"][-1]["id"]
    )
    closing_start = round(decoded["output_start_frame"] / 60 * SR)
    final_closing = {
        "start_sec": closing_start / SR,
        "end_sec": duration,
        "music_zero": float(np.max(abs(bed[closing_start:]), initial=0)) == 0,
        "sfx_zero": float(np.max(abs(fx[closing_start:]), initial=0)) == 0,
    }
    receipt = json.loads((root / "project_receipt.json").read_text(encoding="utf-8"))
    _, actual = video_info(video)
    passed = (
        mix["passed"]
        and actual == assembly["frames"]
        and all(
            w["original_samples_preserved_before_master"]
            and w["added_music_zero"]
            and w["added_effects_zero"]
            for w in protected
        )
        and final_closing["music_zero"]
        and final_closing["sfx_zero"]
    )
    report = {
        "passed": passed,
        "video": str(video),
        "video_sha256": sha(video),
        "camera_parts": len(project["parts"]),
        "original_sound_inserts": protected,
        "closing_voice_only": final_closing,
        "frames": actual,
        "duration_sec": duration,
        "mastering": mix,
        "dialogue_polish": polish,
        "music_continuity": continuity,
        "cues": records,
        "vocal_checks": vocal_report,
        "source_receipt": receipt,
        "synthetic_media_used": False,
    }
    save(root / "delivery-qc.json", report)
    raw = ffmpeg("-i", video, "-af", "ebur128=peak=true", "-f", "null", "-")
    (root / "ebur128.log").write_text(raw, encoding="utf-8")
    if not passed:
        raise PipelineError("Full film protection/closing/frame verification failed")
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--music-manifest", required=True, type=Path)
    args = parser.parse_args()
    print(score_project(args.project, args.music_manifest))
