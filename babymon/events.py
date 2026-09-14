"""Core data types.

Three kinds of thing, deliberately kept distinct:

* ``Frame``       - an image at an instant.
* ``Observation`` - what one detector saw at an instant. Stateless, no
                    interpretation, always carries a confidence.
* ``Alert``       - a decision the rule engine reached over time. The only
                    type that ever reaches a user.

All spatial coordinates are normalised to 0..1 so that configuration does not
depend on camera resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2, normalised


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    ts: float
    seq: int
    source_id: str


@dataclass(frozen=True)
class Observation:
    ts: float
    detector: str
    confidence: float


@dataclass(frozen=True)
class PersonBox(Observation):
    bbox: BBox
    label: str = "unknown"  # "baby" | "adult" | "unknown"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class MotionEnergy(Observation):
    value: float  # 0..1, fraction of zone pixels that changed


@dataclass(frozen=True)
class Pose(Observation):
    # keypoint name -> (x, y, visibility)
    keypoints: dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class CryProbability(Observation):
    value: float  # 0..1


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    AWAKE = "awake"
    CRYING = "crying"
    PRONE = "prone"
    FACE_COVERED = "face_covered"
    ZONE_EXIT = "zone_exit"
    LOST_TRACK = "lost_track"
    DETECTOR_SILENT = "detector_silent"
    SOURCE_LOST = "source_lost"
    CAMERA_MOVED = "camera_moved"


#: Alerts about the monitor itself. Never suppressed - a monitor that has
#: failed must be noisy about it.
HEALTH_ALERTS = frozenset(
    {
        AlertType.LOST_TRACK,
        AlertType.DETECTOR_SILENT,
        AlertType.SOURCE_LOST,
        AlertType.CAMERA_MOVED,
    }
)


@dataclass(frozen=True)
class Alert:
    type: AlertType
    severity: Severity
    started_at: float
    ended_at: float | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
