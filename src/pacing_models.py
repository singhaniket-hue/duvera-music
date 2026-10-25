"""Evidence and decisions are separate; presets are editorial defaults, not perceptual laws."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .models import Contract


class EditEvent(Contract):
    event_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    time_sec: float = Field(ge=0)
    end_sec: float | None = Field(default=None, gt=0)
    kind: Literal["cut", "black_card", "text_enter", "whip", "zoom", "impact", "ui_tick",
                  "shutter", "location_change", "foley", "punch_in", "headline", "data_reveal", "chapter_transition", "riser"]
    confidence: float = Field(ge=0, le=1)
    intensity: float = Field(default=0, ge=0, le=1)
    semantic_salience: float = Field(default=0, ge=0, le=1)
    evidence: str = Field(min_length=1)
    prompt: str | None = None
    location: str | None = None
    eye_anchor: tuple[float, float] | None = None
    zoom_factor: float = Field(default=1.14, ge=1.1, le=1.25)
    required: bool = False

    @model_validator(mode="after")
    def validate_event(self):
        if self.end_sec is not None and self.end_sec <= self.time_sec:
            raise ValueError("Event end must follow its start")
        if self.eye_anchor and any(not 0 <= v <= 1 for v in self.eye_anchor):
            raise ValueError("Eye anchor must use normalized frame coordinates")
        if self.kind == "location_change" and not (self.location and self.prompt):
            raise ValueError("Location events require a location and acoustic prompt")
        if self.kind == "foley" and not self.prompt:
            raise ValueError("Foley requires evidence and an object-specific prompt")
        return self


class EventGraph(Contract):
    schema_version: str = "1.0"
    events: list[EditEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique(self):
        if len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("Event IDs must be unique")
        return self


class SFXInstruction(Contract):
    asset_id: str
    event_id: str
    visual_time_sec: float = Field(ge=0)
    end_sec: float | None = None
    tier: Literal["ambience", "foley", "accent"]
    category: str
    prompt: str
    duration_sec: float = Field(gt=0, le=30)
    generation_sec: float = Field(ge=0.5, le=30)
    av_offset_ms: float = 8
    target_lufs: float
    peak_ceiling_db: float
    j_lead_sec: float = 0
    crossfade_sec: float = 0
    level_mode: Literal["loudness", "sample_peak", "active_rms"] = "loudness"
    target_peak_dbfs: float | None = None
    target_rms_dbfs: float | None = None
    required: bool = False


    @model_validator(mode="after")
    def level_contract(self):
        if self.level_mode == "sample_peak" and self.target_peak_dbfs is None:
            raise ValueError("sample_peak mode requires target_peak_dbfs")
        if self.level_mode == "active_rms" and self.target_rms_dbfs is None:
            raise ValueError("active_rms mode requires target_rms_dbfs")
        return self


class VisualInstruction(Contract):
    event_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    animation_sec: float = 0.18
    zoom_to: float = Field(default=1.14, ge=1.1, le=1.25)
    eye_anchor: tuple[float, float]
    easing: str = "cubic-bezier(0.22, 1, 0.36, 1)"

    @model_validator(mode="after")
    def valid_geometry(self):
        if self.end_sec - self.start_sec < self.animation_sec:
            raise ValueError("Visual interval must contain its zoom animation")
        if any(not 0 <= v <= 1 for v in self.eye_anchor):
            raise ValueError("Eye anchor must use normalized frame coordinates")
        return self


class PacingPlan(Contract):
    effects: list[SFXInstruction] = Field(default_factory=list)
    visuals: list[VisualInstruction] = Field(default_factory=list)
    decisions: list[dict] = Field(default_factory=list)
    pause_review: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
