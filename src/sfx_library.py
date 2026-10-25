"""Persistent acoustic SFX; no oscillator, procedural fallback, or automatic paid retry."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .media import PipelineError, SR, extract_audio, read_audio
from .sfx_engine import crest_sample, shape

PROMPTS = {
    "sub_drop": "Deep cinematic documentary sub-bass boom impact, 40Hz to 80Hz warm analog decay, dark atmospheric thud, no harsh click",
    "paper_slide_pop": "Crisp documentary newspaper clipping slide and soft tactile paper thud, organic foley, warm low-mid body",
    "ui_data_tick": "Subtle mechanical camera aperture click and soft modern UI data tick, crisp 40ms transient, documentary infographic sound",
    "whoosh_soft": "Smooth low-pitched air whoosh transition, soft cinematic cloth and wind sweep, no harsh treble",
    "camera_shutter": "Real professional camera mechanical shutter release, soft tactile double click, organic documentary foley, no voice or musical tone",
}
ALIASES = {"paper_slide": "paper_slide_pop", "ui_tick": "ui_data_tick", "text_pop": "ui_data_tick",
           "whoosh": "whoosh_soft", "riser": "whoosh_soft", "shutter": "camera_shutter"}


def canonical_name(name):
    name = ALIASES.get(name, name)
    if name not in PROMPTS:
        raise PipelineError(f"No acoustic SFX library entry for {name}")
    return name


def analyze_and_shape_sfx_asset(path: Path, name: str) -> dict:
    audio, rate = read_audio(path)
    if rate != SR or audio.shape[1] != 2:
        raise PipelineError("Cached SFX must be stereo 48 kHz PCM16 WAV")
    category = {"paper_slide_pop": "paper_slide", "ui_data_tick": "ui_tick",
                "camera_shutter": "shutter", "whoosh_soft": "whoosh"}.get(name, name)
    shaped = shape(audio, len(audio) / SR, category)
    return {"dominant_peak_time_s": crest_sample(shaped) / SR,
            "raw_peak_time_s": crest_sample(audio) / SR, "sample_rate": SR,
            "duration_seconds": len(audio) / SR,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "landmark": "centred 2ms energy crest refined to a sample peak; remeasured after final DSP"}


class SFXAssetLibrary:
    def __init__(self, directory: Path | None = None):
        self.root = directory or Path(__file__).resolve().parents[1] / "assets" / "sfx_cache"
        self.root.mkdir(parents=True, exist_ok=True)

    def cached(self, name):
        return (self.root / (canonical_name(name) + ".wav")).is_file()

    def get(self, name: str, *, allow_generation=False) -> Path:
        name = canonical_name(name)
        target = self.root / (name + ".wav")
        metadata = self.root / (name + ".json")
        if target.is_file():
            info = analyze_and_shape_sfx_asset(target, name)
            previous = json.loads(metadata.read_text()) if metadata.exists() else {"source": "user_supplied_acoustic_cache"}
            if previous.get("source") == "synthetic":
                raise PipelineError("Synthetic cache assets are not allowed")
            if previous.get("sha256") and previous["sha256"] != info["sha256"]:
                raise PipelineError("SFX cache content changed; review its provenance before use")
            metadata.write_text(json.dumps({**previous, **info}, indent=2))
            return target
        if not allow_generation:
            raise PipelineError(f"Missing acoustic SFX {name}; authorize generation or supply {target}")
        key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not key:
            raise PipelineError("ELEVENLABS_API_KEY is unavailable")
        pending = self.root / (name + ".pending.json")
        try:
            fd = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise PipelineError(f"Uncertain/active SFX generation for {name}; inspect provider history before retrying") from exc
        payload = {"text": PROMPTS[name] + ". No speech, singing or music.",
                   "model_id": "eleven_text_to_sound_v2", "duration_seconds": 1.0, "prompt_influence": 0.45}
        with os.fdopen(fd, "w") as f:
            json.dump({"status": "pending", "request": payload}, f, indent=2)
        raw = self.root / (name + ".mp3")
        temp = self.root / (name + ".partial.wav")
        request = Request("https://api.elevenlabs.io/v1/sound-generation?output_format=mp3_44100_128",
                          data=json.dumps(payload).encode(), headers={"xi-api-key": key, "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=180) as response:
                if "json" in response.headers.get("Content-Type", ""):
                    raise PipelineError("SFX API returned JSON instead of audio")
                request_id = response.headers.get("request-id")
                with raw.open("wb") as f:
                    while block := response.read(1024 * 1024):
                        f.write(block)
            extract_audio(raw, temp)
            info = analyze_and_shape_sfx_asset(temp, name)
            temp.replace(target)
            metadata.write_text(json.dumps({**info, "source": "elevenlabs_sound_generation",
                "request": payload, "request_id": request_id, "synthetic_generator_used": False}, indent=2))
            pending.unlink()
            return target
        except (HTTPError, URLError, TimeoutError) as exc:
            raise PipelineError(f"Acoustic SFX generation failed ({type(exc).__name__}); no automatic retry; pending record retained") from None
