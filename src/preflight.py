import importlib.util
import os
import platform
import shutil
from pathlib import Path


def check_environment(profile: Path = Path("chrome_google_flow_profile")) -> dict:
    return {
        "python": platform.python_version(), "platform": platform.system(),
        "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"),
        "has_elevenlabs_api_key": bool(os.getenv("ELEVENLABS_API_KEY", "").strip()),
        "elevenlabs_cli_detected": bool(shutil.which("elevenlabs")),
        "google_flow_profile_exists": profile.exists(),
        "google_flow_authentication": "not_verified",
        "llm_configured": bool(os.getenv("DIRECTOR_LLM_URL") and os.getenv("DIRECTOR_LLM_MODEL")),
        "optional_dependencies": {x: importlib.util.find_spec(x) is not None for x in
                                  ("faster_whisper", "librosa", "playwright", "opentimelineio", "demucs", "torch", "torchaudio", "cv2", "resemblyzer")},
        "note": "CLI/plugin presence and profile files are not proof of provider authorization.",
    }
