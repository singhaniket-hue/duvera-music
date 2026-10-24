"""Official music and SFX endpoints; no automatic retry of billable requests."""
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..media import PipelineError
from ..music_policy import instrumental_prompt


class ElevenLabsProvider:
    def __init__(self):
        self.key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not self.key:
            raise PipelineError("Export ELEVENLABS_API_KEY; keys are never accepted in CLI arguments")

    @staticmethod
    def payload(prompt: str, duration: float, kind: str) -> tuple[str, dict]:
        if kind == "music":
            if not 3 <= duration <= 600:
                raise PipelineError("ElevenLabs music length must be 3–600 seconds")
            return "/music?output_format=mp3_48000_192", {
                "prompt": instrumental_prompt(prompt), "music_length_ms": round(duration * 1000),
                "model_id": "music_v2", "force_instrumental": True,
            }
        if not 0.5 <= duration <= 30:
            raise PipelineError("SFX length must be 0.5–30 seconds")
        return "/sound-generation?output_format=mp3_44100_128", {
            "text": prompt, "duration_seconds": duration,
            "model_id": "eleven_text_to_sound_v2", "prompt_influence": 0.3,
        }

    def generate(self, prompt: str, duration: float, kind: str, destination: Path):
        endpoint, payload = self.payload(prompt, duration, kind)
        request = Request("https://api.elevenlabs.io/v1" + endpoint,
                          data=json.dumps(payload).encode(), headers={
                              "xi-api-key": self.key, "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=600) as response, destination.open("wb") as f:
                if "json" in response.headers.get("Content-Type", ""):
                    raise PipelineError("Provider returned JSON instead of audio")
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
        except HTTPError as exc:
            raise PipelineError(f"ElevenLabs returned HTTP {exc.code}; no automatic retry") from None
        except (URLError, TimeoutError) as exc:
            raise PipelineError("ElevenLabs transport failure; check provider history before retrying") from exc
