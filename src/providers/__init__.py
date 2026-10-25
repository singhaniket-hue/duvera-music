"""Provider adapters and durable generation ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from ..director import music_prompt
from ..media import PipelineError, extract_audio, probe
from ..models import RunConfig, SpottingSheet
from ..narrative import fully_silent
from ..music_policy import instrumental_prompt


class AudioProvider(Protocol):
    def generate(self, prompt: str, duration: float, kind: str, destination: Path) -> None: ...


def read_manifest(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = {"music": {}, "sfx": {}, "downbeats": raw.get("downbeats", {}), "loops": raw.get("loops", {})}
    for kind in ("music", "sfx"):
        for key, value in raw.get(kind, {}).items():
            p = Path(value).expanduser()
            result[kind][key] = (path.parent / p).resolve() if not p.is_absolute() else p
            if not result[kind][key].is_file():
                raise PipelineError(f"Missing asset for {key}: {p}")
    return result


class AssetStore:
    """A persisted pending call is never replayed automatically after an uncertain failure."""

    def __init__(self, root: Path, config: RunConfig, manifest: Path | None, provider: AudioProvider | None):
        self.root, self.config, self.provider = root, config, provider
        root.mkdir(parents=True, exist_ok=True)
        self.assets = read_manifest(manifest) if manifest else {"music": {}, "sfx": {}, "downbeats": {}, "loops": {}}
        self.ledger_path = root / "generation_ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}
        self.procedural = {}
        self.acoustic = {}

    def _save(self):
        temp = self.ledger_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.ledger, indent=2))
        temp.replace(self.ledger_path)

    def get(self, cue_id: str, prompt: str, duration: float, kind: str, loop: bool = False) -> Path:
        if kind == "music":
            prompt = instrumental_prompt(prompt)
        key = f"{cue_id}_{kind}"
        output = self.root / f"{key}.wav"
        if cue_id in self.assets[kind]:
            extract_audio(self.assets[kind][cue_id], output)
            return output
        if kind == "sfx" and cue_id in self.acoustic:
            from ..sfx_library import SFXAssetLibrary
            source = SFXAssetLibrary(self.config.sfx_cache_dir).get(
                self.acoustic[cue_id].category, allow_generation=self.config.allow_paid_generation)
            extract_audio(source, output)
            return output
        if self.provider is None:
            raise PipelineError(f"Manifest is missing {kind} for {cue_id}")
        fingerprint = hashlib.sha256(json.dumps([prompt, duration, kind, loop]).encode()).hexdigest()
        entry = self.ledger.get(key)
        if entry:
            if entry["fingerprint"] != fingerprint:
                raise PipelineError("Generation plan changed; use a new output directory")
            if entry["status"] == "complete" and output.exists():
                return output
            raise PipelineError(f"Uncertain prior generation: {key}. Inspect generation_ledger.json and provider "
                                "billing/history before removing that entry to authorize a retry.")
        if sum(e.get("mode") != "procedural" for e in self.ledger.values()) >= self.config.max_requests:
            raise PipelineError("Generation request budget exhausted")
        if sum(e["seconds"] for e in self.ledger.values()) + duration > self.config.max_generated_seconds:
            raise PipelineError("Generation duration budget exhausted")
        self.ledger[key] = {"fingerprint": fingerprint, "status": "pending", "seconds": duration}
        self._save()
        raw = self.root / f"{key}.mp3"
        if loop:
            self.provider.generate(prompt, duration, kind, raw, loop=True)
        else:
            self.provider.generate(prompt, duration, kind, raw)
        extract_audio(raw, output)
        if float(probe(output)["format"]["duration"]) <= 0:
            raise PipelineError("Provider returned empty audio")
        self.ledger[key]["status"] = "complete"
        self._save()
        return output


def generation_jobs(sheet: SpottingSheet, pacing=None, locked_music=False):
    if locked_music:
        if pacing:
            for effect in pacing.effects:
                yield SimpleNamespace(cue_id=effect.asset_id, loop=effect.tier == "ambience"), "sfx", effect.generation_sec, effect.prompt
        return
    for cue in sheet.cues:
        if fully_silent(cue.start_sec, cue.end_sec, sheet.silence_windows):
            continue
        if not cue.reuse_music_from:
            yield cue, "music", cue.generation_seconds or max(3, cue.end_sec - cue.start_sec), music_prompt(cue)
        if cue.bridge_sfx_prompt:
            yield cue, "sfx", max(0.5, min(2, cue.end_sec - cue.start_sec)), cue.bridge_sfx_prompt
    if pacing:
        for effect in pacing.effects:
            yield SimpleNamespace(cue_id=effect.asset_id, loop=effect.tier == "ambience"), "sfx", effect.generation_sec, effect.prompt


def validate_manifest_for_sheet(path: Path, sheet: SpottingSheet, pacing=None, locked_music=False, procedural=None):
    manifest = read_manifest(path)
    for cue, kind, _, _ in generation_jobs(sheet, pacing, locked_music):
        if cue.cue_id not in manifest[kind]:
            if kind == "sfx" and procedural and cue.cue_id in procedural:
                continue
            raise PipelineError(f"Manifest {path.name} missing {kind}/{cue.cue_id}")
        probe(manifest[kind][cue.cue_id])
    return manifest
