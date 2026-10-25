"""Real multi-part documentary assembly with pause/resume source-sound inserts.

Project files and media stay outside the repository. All segment boundaries use
integer 60fps frames and 48kHz samples; external inserts own their audio windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .archival import ArchivalInsertWindow
from .documentary_pipeline import prepare_documentary
from .documentary_schema import TimelineMap
from .media import SR, PipelineError, extract_audio, ffmpeg, probe, read_audio, write_audio
from .models import RunConfig, Transcript
from .pacing_models import EditEvent, EventGraph


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def video_info(path):
    stream = next(s for s in probe(Path(path))["streams"] if s["codec_type"] == "video")
    return stream, int(stream["nb_frames"])


def prepare_part(spec, root):
    name = spec["id"]
    target = root / "parts" / name
    target.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
        + Path(spec["transcript"]).read_bytes()
        + Path(spec["plan"]).read_bytes()
        + b"".join(
            Path(__file__).with_name(name).read_bytes()
            for name in (
                "video_editor_engine.py",
                "html_overlay_renderer.py",
                "studio_grade.py",
                "media_cutaway.py",
                "documentary_pipeline.py",
                "retake_cutter.py",
            )
        )
    ).hexdigest()
    marker = target / "part_receipt.json"
    if marker.exists():
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        if receipt["signature"] != signature:
            raise PipelineError("Part plan changed; use a new project output folder")
        if receipt["picture_sha256"] == sha(receipt["picture"]) and receipt["source_sha256"] == sha(
            spec["source"]
        ):
            return receipt
        raise PipelineError("Rendered part hash changed")
    source = Path(spec["source"])
    if spec.get("reuse_render_dir"):
        cached = Path(spec["reuse_render_dir"])
        verification = json.loads(Path(spec["reuse_delivery_qc"]).read_text(encoding="utf-8"))
        if not verification["passed"] or verification["real_source_sha256"] != sha(source):
            raise PipelineError("Reused real render does not match its verified original source")
        render_marker = json.loads((cached / "documentary_video.render.json").read_text())
        picture = cached / "documentary_video.mp4"
        if sha(picture) != render_marker["video_sha256"]:
            raise PipelineError("Reused picture hash changed")
        for filename in (
            "timeline_map.json",
            "transcript.json",
            "documentary_visual_plan.json",
            "retake_analysis.json",
            "documentary_render_qc.json",
        ):
            shutil.copyfile(cached / filename, target / filename)
        picture = str(picture)
        dialogue = str(cached / "dialogue_edited.wav")
        graph = json.loads((cached / "event_graph.json").read_text(encoding="utf-8"))
        report = json.loads((cached / "documentary_render_qc.json").read_text(encoding="utf-8"))
        silence = json.loads((cached / "spotting_sheet.json").read_text(encoding="utf-8"))["silence_windows"]
    else:
        config = RunConfig(
            preset="social_commentary",
            video_path=source,
            transcript_path=Path(spec["transcript"]),
            documentary_plan_path=Path(spec["plan"]),
            output_dir=target,
            mode="assets_and_script",
            provider="elevenlabs",
            retake_autocut=True,
            studio_commentary_grade=True,
            language="hinglish",
        )
        transcript = Transcript.model_validate_json(config.transcript_path.read_text(encoding="utf-8"))
        stream, _ = video_info(source)
        duration = int(stream["nb_frames"]) / 60
        raw = target / "dialogue.wav"
        extract_audio(source, raw, duration)
        result = prepare_documentary(config, transcript, duration, raw, [], False)
        picture = str(result["config"].video_path)
        dialogue = str(result["dialogue"])
        graph = result["graph"].model_dump(mode="json")
        report = result["report"]
        silence = [s.model_dump() for s in result["silence"]]
    _, frames = video_info(picture)
    receipt = {
        "id": name,
        "signature": signature,
        "source": str(source),
        "source_sha256": sha(source),
        "picture": picture,
        "picture_sha256": sha(picture),
        "dialogue": dialogue,
        "frames": frames,
        "events": graph["events"],
        "silence_windows": silence,
        "report": report,
    }
    save(marker, receipt)
    print("PART COMPLETE", name, frames, "frames", flush=True)
    return receipt


def assembly_segments(receipts, parts, insertions):
    """Pause an edited camera clock, insert original source sound, then resume it."""
    result = []
    cursor = 0
    for spec, receipt in zip(parts, receipts):
        mapping = TimelineMap.model_validate_json((Path(receipt["root"]) / "timeline_map.json").read_text())
        local = 0
        items = sorted(
            (i for i in insertions if i["part_id"] == spec["id"]), key=lambda i: i["at_source_sec"]
        )
        for insertion in items:
            boundary = round(mapping.time(insertion["at_source_sec"]) * 60)
            if boundary <= local or boundary >= receipt["frames"]:
                raise PipelineError("Invalid pause/resume insert boundary")
            result.append(
                {
                    "kind": "camera",
                    "part_id": spec["id"],
                    "source_start_frame": local,
                    "source_end_frame": boundary,
                    "output_start_frame": cursor,
                    "picture": receipt["picture"],
                    "dialogue": receipt["dialogue"],
                }
            )
            cursor += boundary - local
            count = round(insertion["duration_sec"] * 60)
            result.append({"kind": "archive", "output_start_frame": cursor, "frames": count, **insertion})
            cursor += count
            local = boundary
        result.append(
            {
                "kind": "camera",
                "part_id": spec["id"],
                "source_start_frame": local,
                "source_end_frame": receipt["frames"],
                "output_start_frame": cursor,
                "picture": receipt["picture"],
                "dialogue": receipt["dialogue"],
            }
        )
        cursor += receipt["frames"] - local
    return result, cursor


def remap_part_time(part_id, time, segments):
    frame = time * 60
    matching = [s for s in segments if s["kind"] == "camera" and s["part_id"] == part_id]
    for segment in matching:
        if frame < segment["source_end_frame"]:
            return (segment["output_start_frame"] + max(0, frame - segment["source_start_frame"])) / 60
    segment = matching[-1]
    return (segment["output_start_frame"] + segment["source_end_frame"] - segment["source_start_frame"]) / 60


def render_assembly(segments, frames, root):
    from .media_cutaway import credit_png

    pieces = root / "assembly"
    pieces.mkdir(exist_ok=True)
    files = []
    sound = np.zeros((round(frames / 60 * SR), 2), np.float32)
    archives = []
    for number, segment in enumerate(segments):
        target = pieces / f"{number:03}.mp4"
        start = segment["output_start_frame"] * 800
        if segment["kind"] == "camera":
            first, last = segment["source_start_frame"], segment["source_end_frame"]
            count = last - first
            voice, _ = read_audio(Path(segment["dialogue"]))
            sound[start : start + count * 800] = voice[first * 800 : last * 800]
            _, whole = video_info(segment["picture"])
            if not target.exists():
                if first == 0 and last == whole:
                    ffmpeg("-i", segment["picture"], "-an", "-c:v", "copy", target)
                else:
                    ffmpeg(
                        "-i",
                        segment["picture"],
                        "-vf",
                        f"trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-threads",
                        "4",
                        "-pix_fmt",
                        "yuv420p",
                        target,
                    )
        else:
            count = segment["frames"]
            original = Path(segment["source"])
            if sha(original) != segment["source_sha256"]:
                raise PipelineError("Original-sound insert changed")
            offset = segment["source_in_sec"]
            label = credit_png(segment["credit"], 1920, 1080, root / "archive_credits")
            if not target.exists():
                filt = (
                    "[0:v]fps=60,split=2[b][f];[b]scale=1920:1080:force_original_aspect_ratio=increase,"
                    "crop=1920:1080,gblur=sigma=22,eq=brightness=-0.15[bg];"
                    "[f]scale=1920:1080:force_original_aspect_ratio=decrease[fg];"
                    "[bg][fg]overlay=(W-w)/2:(H-h)/2[image];[image][1:v]overlay=0:0[v]"
                )
                ffmpeg(
                    "-ss",
                    offset,
                    "-i",
                    original,
                    "-loop",
                    "1",
                    "-i",
                    label,
                    "-t",
                    count / 60,
                    "-filter_complex",
                    filt,
                    "-map",
                    "[v]",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-threads",
                    "4",
                    "-pix_fmt",
                    "yuv420p",
                    "-frames:v",
                    count,
                    target,
                )
            wav = pieces / f"{number:03}-original.wav"
            ffmpeg(
                "-ss",
                offset,
                "-i",
                original,
                "-t",
                count / 60,
                "-vn",
                "-ac",
                "2",
                "-ar",
                SR,
                "-c:a",
                "pcm_s16le",
                wav,
            )
            source_audio, _ = read_audio(wav)
            if len(source_audio) < count * 800:
                raise PipelineError("Original source sound is shorter than its planned insert")
            sound[start : start + count * 800] = source_audio[: count * 800]
            archives.append(
                ArchivalInsertWindow(
                    start_sec=start / SR,
                    end_sec=(start + count * 800) / SR,
                    source=segment["credit"],
                    evidence="Reviewed original-sound insert; narration clock paused",
                )
            )
        _, actual = video_info(target)
        if actual != count:
            raise PipelineError("Assembly piece frame count differs from exact AV clock")
        files.append(target)
    concat = pieces / "concat.txt"
    concat.write_text(
        "".join("file '" + str(p.resolve()).replace("\\", "/") + "'\n" for p in files), encoding="utf-8"
    )
    picture = root / "full_picture.mp4"
    ffmpeg(
        "-f", "concat", "-safe", "0", "-i", concat, "-an", "-c:v", "copy", "-movflags", "+faststart", picture
    )
    _, actual = video_info(picture)
    if actual != frames:
        raise PipelineError("Full film picture lost frames during assembly")
    write_audio(root / "dialogue_assembled.wav", sound)
    return picture, archives


def prepare_project(path, jobs=2):
    project = json.loads(Path(path).read_text(encoding="utf-8"))
    root = Path(project["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        receipts = list(pool.map(lambda s: prepare_part(s, root), project["parts"]))
    for receipt in receipts:
        receipt["root"] = str(root / "parts" / receipt["id"])
    segments, frames = assembly_segments(receipts, project["parts"], project.get("insertions", []))
    save(root / "assembly_plan.json", {"frames": frames, "fps": 60, "segments": segments})
    picture, archives = render_assembly(segments, frames, root)
    events = []
    silence = []
    for receipt in receipts:
        for event in receipt["events"]:
            event = {
                **event,
                "event_id": receipt["id"] + "_" + event["event_id"],
                "time_sec": remap_part_time(receipt["id"], event["time_sec"], segments),
            }
            if event.get("end_sec") is not None:
                event["end_sec"] = remap_part_time(receipt["id"], event["end_sec"], segments)
            if event.get("documentary_card_id"):
                event["documentary_card_id"] = receipt["id"] + "_" + event["documentary_card_id"]
            events.append(EditEvent(**event))
        for window in receipt["silence_windows"]:
            silence.append(
                {
                    **window,
                    "start_sec": remap_part_time(receipt["id"], window["start_sec"], segments),
                    "end_sec": remap_part_time(receipt["id"], window["end_sec"], segments),
                    "accent_event_ids": [receipt["id"] + "_" + e for e in window["accent_event_ids"]],
                }
            )
    save(root / "event_graph.json", EventGraph(events=events).model_dump(mode="json"))
    save(root / "silence_windows.json", silence)
    save(root / "archival_windows.json", {"windows": [a.model_dump() for a in archives]})
    save(
        root / "project_receipt.json",
        {
            "picture": str(picture),
            "picture_sha256": sha(picture),
            "frames": frames,
            "duration_sec": frames / 60,
            "parts": receipts,
            "original_sound_inserts": len(archives),
            "synthetic_media_used": False,
        },
    )
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=2, choices=(1, 2, 3))
    args = parser.parse_args()
    print(prepare_project(args.project, args.jobs))
