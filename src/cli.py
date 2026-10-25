"""Nonblocking agent CLI and optional TTY onboarding wizard."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .media import PipelineError
from .models import RunConfig
from .pipeline import execute
from .preflight import check_environment

PATHS = {"video_path", "script_path", "transcript_path", "spotting_sheet_path", "event_graph_path", "locked_music_path", "output_dir",
         "archival_windows_path", "narrator_reference_path",
         "local_assets", "google_flow_assets", "elevenlabs_assets", "google_flow_browser_config", "chrome_profile_dir"}


def parser():
    p = argparse.ArgumentParser(description="Duvera dialogue-aware music director")
    p.add_argument("--check-env", action="store_true")
    p.add_argument("--json-config", type=Path)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="Analyze/plan without generating or rendering")
    for flag, dest in [("video", "video_path"), ("script", "script_path"), ("transcript", "transcript_path"),
                       ("spotting-sheet", "spotting_sheet_path"), ("output-dir", "output_dir"),
                       ("event-graph", "event_graph_path"),
                       ("locked-music", "locked_music_path"),
                       ("archival-windows", "archival_windows_path"), ("narrator-reference", "narrator_reference_path"),
                       ("local-assets", "local_assets"), ("google-flow-assets", "google_flow_assets"),
                       ("elevenlabs-assets", "elevenlabs_assets"), ("google-flow-browser-config", "google_flow_browser_config"),
                       ("chrome-profile-dir", "chrome_profile_dir")]:
        p.add_argument("--" + flag, dest=dest, type=Path)
    p.add_argument("--mode", choices=["script_only", "assets_and_script", "final_video"])
    p.add_argument("--lang", dest="language", choices=["english", "hinglish", "hindi"])
    p.add_argument("--provider", choices=["none", "local", "elevenlabs", "google_flow", "both_compare"])
    p.add_argument("--planner", choices=["rules", "llm"])
    p.add_argument("--preset", choices=["default", "social_commentary"])
    p.add_argument("--archival-detection", choices=["off", "reference_voice"])
    p.add_argument("--speaker-mismatch-threshold", type=float)
    p.add_argument("--speaker-return-threshold", type=float)
    vocal_chain = p.add_mutually_exclusive_group()
    vocal_chain.add_argument("--dialogue-polish", action="store_true", default=None)
    vocal_chain.add_argument("--no-dialogue-polish", dest="dialogue_polish", action="store_false", default=None)
    p.add_argument("--allow-paid-generation", action="store_true", default=None)
    for name in ("max-generated-seconds", "target-lufs", "true-peak-db", "cue-seconds", "estimated-wpm"):
        p.add_argument("--" + name, type=float)
    p.add_argument("--max-requests", type=int)
    p.add_argument("--fps", type=int)
    p.add_argument("--asr-model")
    p.add_argument("--asr-device")
    p.add_argument("--no-visual-analysis", dest="analyze_visuals", action="store_false", default=None)
    p.add_argument("--visual-sample-fps", type=int)
    p.add_argument("--black-luma-threshold", type=float)
    p.add_argument("--black-pixel-ratio", type=float)
    p.add_argument("--black-min-duration-sec", type=float)
    p.add_argument("--beats-per-bar", type=int)
    p.add_argument("--loop-crossfade-ms", type=float)
    p.add_argument("--vocal-policy", choices=["reject", "strip", "off"])
    p.add_argument("--vocal-threshold-dbfs", type=float)
    p.add_argument("--vocal-model")
    p.add_argument("--vocal-device", choices=["cpu", "cuda", "mps"])
    outro = p.add_mutually_exclusive_group()
    outro.add_argument("--outro-music-return", dest="outro_music_return", action="store_true", default=None)
    outro.add_argument("--no-outro-music-return", dest="outro_music_return", action="store_false", default=None)
    p.add_argument("--outro-return-sec", type=float)
    p.add_argument("--sfx-mode", choices=["none", "accents_only", "full_3_tier"])
    p.add_argument("--sfx-mix-profile", choices=["research", "audible"])
    p.add_argument("--locked-music-sha256")
    p.add_argument("--max-music-gap-sec", type=float)
    p.add_argument("--sfx-silence-policy", choices=["protect", "accents"])
    p.add_argument("--visual-pacing", choices=["off", "plan", "render"])
    p.add_argument("--no-pacing-detection", dest="pacing_detect_video", action="store_false", default=None)
    p.add_argument("--accent-cooldown-sec", type=float)
    p.add_argument("--major-cooldown-sec", type=float)
    p.add_argument("--accent-salience-threshold", type=float)
    return p


def wizard() -> dict:
    def choice(prompt, values):
        while True:
            print(prompt)
            for i, value in enumerate(values, 1):
                print(f"  {i}. {value}")
            answer = input("> ").strip()
            if answer.isdigit() and 1 <= int(answer) <= len(values):
                return values[int(answer) - 1]
            if answer in values:
                return answer
    mode = choice("Output:", ["script_only", "assets_and_script", "final_video"])
    language = choice("Spoken language:", ["english", "hinglish", "hindi"])
    data = {"mode": mode, "language": language}
    data["sfx_mode"] = choice("Sound design:", ["none", "accents_only", "full_3_tier"])
    for key, label in [("video_path", "Video/audio path"), ("script_path", "Script path"),
                       ("transcript_path", "Timestamped transcript JSON path")]:
        value = input(label + " (Enter to skip): ").strip().strip('"')
        if value:
            data[key] = value
    if mode != "script_only":
        data["provider"] = choice("Music provider:", ["local", "elevenlabs", "google_flow", "both_compare"])
        for key, label in [("local_assets", "Local assets manifest"),
                           ("elevenlabs_assets", "ElevenLabs import manifest (optional)"),
                           ("google_flow_assets", "Google Flow import manifest (optional)"),
                           ("google_flow_browser_config", "Google browser config (optional)")]:
            relevant = (key == "local_assets" and data["provider"] == "local" or
                        key.startswith("elevenlabs") and data["provider"] in ("elevenlabs", "both_compare") or
                        key.startswith("google") and data["provider"] in ("google_flow", "both_compare"))
            if relevant:
                value = input(label + ": ").strip().strip('"')
                if value:
                    data[key] = value
        if data["provider"] != "local":
            data["allow_paid_generation"] = input("Authorize generation using account credits? [yes/no]: ").strip() == "yes"
    data["output_dir"] = input("Output directory [agent_output]: ").strip() or "agent_output"
    return data


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.check_env:
        print(json.dumps(check_environment(args.chrome_profile_dir or Path("chrome_google_flow_profile")), indent=2))
        return 0
    supplied = {k: v for k, v in vars(args).items() if v is not None and k not in
                {"check_env", "json_config", "interactive", "plan_only"}}
    if args.interactive or (not supplied and args.json_config is None and sys.stdin.isatty()):
        if not sys.stdin.isatty():
            p.error("--interactive requires a terminal; agents should use flags or --json-config")
        supplied = {**wizard(), **supplied}
    try:
        data = {}
        if args.json_config:
            data = json.loads(args.json_config.read_text(encoding="utf-8"))
            for key in PATHS:
                if data.get(key):
                    path = Path(data[key]).expanduser()
                    data[key] = path if path.is_absolute() else args.json_config.resolve().parent / path
        data.update(supplied)
        config = RunConfig.model_validate(data)
        for field in PATHS:
            value = getattr(config, field)
            if value is not None:
                setattr(config, field, value.expanduser().resolve())
        summary = execute(config, args.plan_only)
        print(json.dumps(summary, indent=2))
        return 3 if summary["status"] == "qc_failed" else 0
    except (ValidationError, PipelineError, ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('{"status":"cancelled"}', file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
