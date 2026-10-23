"""Validated interchange contracts. All timeline units are seconds."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Segment(Contract):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Segment end must follow start")
        return self


class Transcript(Contract):
    segments: list[Segment] = Field(min_length=1)
    timing_source: Literal["asr", "supplied", "estimated"] = "supplied"

    @model_validator(mode="after")
    def ordered(self):
        for a, b in zip(self.segments, self.segments[1:]):
            if b.start < a.end - 0.001:
                raise ValueError("Transcript segments must be sorted and non-overlapping")
        return self


class SilenceWindow(Contract):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    reason: str

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("Silence window end must follow start")
        return self


class MusicCue(Contract):
    cue_id: str = Field(pattern=r"^cue_[0-9]{3}$")
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    narrative_purpose: str
    speech_syllables_per_sec: float = Field(ge=0)
    prosodic_pulse_hz: float | None = Field(default=None, ge=0)
    target_bpm: int = Field(ge=50, le=160)
    mood_descriptors: list[str]
    allowed_instruments: list[str]
    depth_profile: Literal["barely_behind", "background", "cinematic_distant", "montage_foreground"]
    target_lu_gap: float = Field(ge=0, le=24)
    exit_type: Literal["transient_stop", "crossfade", "riser_bridge"] = "crossfade"
    bridge_sfx_prompt: str | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("Cue end must follow start")
        return self


class SpottingSheet(Contract):
    schema_version: str = "1.0"
    video_duration_sec: float = Field(gt=0)
    dialogue_integrated_lufs: float | None = None
    timing_source: Literal["asr", "supplied", "estimated"]
    planner: str = "rules"
    cues: list[MusicCue] = Field(min_length=1)
    silence_windows: list[SilenceWindow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timeline(self):
        if len({c.cue_id for c in self.cues}) != len(self.cues):
            raise ValueError("Cue IDs must be unique")
        for i, c in enumerate(self.cues):
            if c.end_sec > self.video_duration_sec + 0.001:
                raise ValueError("Cue exceeds media duration")
            if i and c.start_sec < self.cues[i - 1].end_sec - 0.001:
                raise ValueError("Cues must be sorted and non-overlapping")
        for s in self.silence_windows:
            if s.end_sec > self.video_duration_sec + 0.001:
                raise ValueError("Silence window exceeds media duration")
        return self


class RunConfig(Contract):
    video_path: Path | None = None
    script_path: Path | None = None
    transcript_path: Path | None = None
    spotting_sheet_path: Path | None = None
    output_dir: Path = Path("agent_output")
    mode: Literal["script_only", "assets_and_script", "final_video"] = "script_only"
    language: Literal["english", "hinglish", "hindi"] = "english"
    provider: Literal["none", "local", "elevenlabs", "google_flow", "both_compare"] = "none"
    planner: Literal["rules", "llm"] = "rules"
    local_assets: Path | None = None
    google_flow_assets: Path | None = None
    elevenlabs_assets: Path | None = None
    google_flow_browser_config: Path | None = None
    chrome_profile_dir: Path = Path("chrome_google_flow_profile")
    allow_paid_generation: bool = False
    max_generated_seconds: float = Field(default=600, gt=0, le=86400)
    max_requests: int = Field(default=30, ge=1, le=1000)
    target_lufs: float = Field(default=-15, ge=-24, le=-9)
    true_peak_db: float = Field(default=-1.5, ge=-6, le=-1)
    cue_seconds: float = Field(default=35, ge=3, le=120)
    estimated_wpm: float = Field(default=140, ge=60, le=260)
    asr_model: str = "small"
    asr_device: str = "cpu"
    fps: int = Field(default=30, ge=1, le=120)

    @model_validator(mode="after")
    def coherent(self):
        if not (self.video_path or self.script_path or self.transcript_path):
            raise ValueError("Supply video/audio, a script, or a timestamped transcript")
        if self.mode != "script_only" and self.provider == "none":
            raise ValueError("Asset and video outputs require a provider")
        if self.mode == "final_video" and not self.video_path:
            raise ValueError("final_video requires video_path")
        if self.provider == "local" and self.mode != "script_only" and not self.local_assets:
            raise ValueError("local provider requires local_assets manifest")
        return self
