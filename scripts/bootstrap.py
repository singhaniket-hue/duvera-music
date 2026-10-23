"""Idempotent cross-platform setup; system package installation is explicit."""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extras", default="", help="Comma-separated: asr,mir,browser,timeline,dev")
    parser.add_argument("--install-ffmpeg", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required")
    allowed = {"asr", "mir", "browser", "timeline", "dev"}
    extras = [s.strip() for s in args.extras.split(",") if s.strip()]
    if set(extras) - allowed:
        parser.error("Unknown extra")
    if not all(shutil.which(x) for x in ("ffmpeg", "ffprobe")):
        if args.install_ffmpeg:
            system = platform.system()
            if system == "Darwin" and shutil.which("brew"):
                cmd = ["brew", "install", "ffmpeg"]
            elif system == "Windows" and shutil.which("winget"):
                cmd = ["winget", "install", "--id", "Gyan.FFmpeg", "-e"]
            elif shutil.which("apt-get") and hasattr(os, "geteuid") and os.geteuid() == 0:
                subprocess.run(["apt-get", "update"], check=True)
                cmd = ["apt-get", "install", "-y", "ffmpeg"]
            else:
                parser.error("Install FFmpeg with your system package manager, then rerun")
            subprocess.run(cmd, check=True)
        else:
            print("FFmpeg/ffprobe missing. Install separately or use --install-ffmpeg. Script-only mode still works.")
    env = ROOT / ".venv"
    if not env.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    target = str(ROOT) + ("[" + ",".join(extras) + "]" if extras else "")
    subprocess.run([str(python), "-m", "pip", "install", "-e", target], check=True)
    if "browser" in extras:
        subprocess.run([str(python), "-m", "playwright", "install", "chromium"], check=True)
    subprocess.run([str(python), "-m", "src.cli", "--check-env"], cwd=ROOT, check=True)
    print(f"Ready. Python: {python}")


if __name__ == "__main__":
    main()
