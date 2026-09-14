"""Configuration. Invalid config fails loudly at startup, not at 3am."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ZoneConfig(BaseModel):
    name: str
    polygon: list[tuple[float, float]] = Field(min_length=3)

    @field_validator("polygon")
    @classmethod
    def _normalised(
        cls, v: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"polygon point ({x}, {y}) is outside 0..1; "
                    "zone coordinates are normalised"
                )
        return v


class SourceConfig(BaseModel):
    kind: Literal["webcam", "file"] = "webcam"
    device: int = 0
    path: str | None = None
    realtime: bool = True
    source_id: str = "cam0"

    @model_validator(mode="after")
    def _path_required_for_file(self) -> "SourceConfig":
        if self.kind == "file" and not self.path:
            raise ValueError("source.path is required when source.kind is 'file'")
        return self


class EventRuleConfig(BaseModel):
    enter_threshold: float = 0.5
    exit_threshold: float = 0.3
    min_duration_s: float = 2.0
    exit_duration_s: float = 3.0
    cooldown_s: float = 60.0

    @model_validator(mode="after")
    def _hysteresis_ordered(self) -> "EventRuleConfig":
        if self.exit_threshold > self.enter_threshold:
            raise ValueError(
                "exit_threshold must not exceed enter_threshold; "
                "hysteresis requires enter >= exit"
            )
        return self


class RulesConfig(BaseModel):
    awake: EventRuleConfig = Field(
        default_factory=lambda: EventRuleConfig(
            enter_threshold=0.02, exit_threshold=0.005,
            min_duration_s=2.0, exit_duration_s=10.0, cooldown_s=120.0,
        )
    )
    zone_exit: EventRuleConfig = Field(
        default_factory=lambda: EventRuleConfig(
            enter_threshold=0.5, exit_threshold=0.5,
            min_duration_s=2.0, exit_duration_s=3.0, cooldown_s=60.0,
        )
    )
    # Adult presence is not a thresholded signal, it is a recency question:
    # how long since we last saw an adult, and did we see one long enough to
    # believe it. Hence plain durations rather than an EventRuleConfig.
    adult_min_duration_s: float = 1.0
    adult_hold_s: float = 5.0
    lost_track_after_s: float = 60.0
    detector_silent_after_s: float = 15.0
    # A health condition that persists must keep saying so. Latching on the
    # first alert meant a blanket over the camera produced exactly one
    # notification and then silence all night - and one missed notification
    # left the monitor indistinguishable from calm. These are re-alert
    # intervals, not cooldowns on an episode: the condition is still true.
    lost_track_realert_interval_s: float = 300.0
    detector_silent_realert_interval_s: float = 300.0
    reorder_window_s: float = 0.25


class DetectorsConfig(BaseModel):
    person_conf_threshold: float = 0.4
    person_stride: int = 1
    baby_max_area: float = 0.25
    motion_stride: int = 1
    motion_pixel_threshold: int = 25
    model_path: str = "yolo11n.pt"


class StorageConfig(BaseModel):
    db_path: str = "data/events.db"
    snapshot_dir: str = "data/snapshots"
    retention_days: int = 7


class Config(BaseModel):
    source: SourceConfig = Field(default_factory=SourceConfig)
    zones: list[ZoneConfig] = Field(default_factory=list)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    detectors: DetectorsConfig = Field(default_factory=DetectorsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def zone(self, name: str) -> ZoneConfig | None:
        return next((z for z in self.zones if z.name == name), None)
