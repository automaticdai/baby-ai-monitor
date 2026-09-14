"""Region-of-interest polygons in normalised coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from babymon.config import ZoneConfig


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        """Coerce polygon to immutable tuple of tuples."""
        if not isinstance(self.polygon, tuple):
            object.__setattr__(
                self, "polygon", tuple((float(x), float(y)) for x, y in self.polygon)
            )

    @classmethod
    def from_config(cls, cfg: ZoneConfig) -> "Zone":
        return cls(name=cfg.name, polygon=cfg.polygon)

    def contains(self, point: tuple[float, float]) -> bool:
        """Ray-casting point-in-polygon.

        Points exactly on an edge are not guaranteed to be inside; callers
        should not depend on boundary behaviour, and the rule layer's
        hysteresis makes single-pixel boundary cases irrelevant anyway.
        """
        x, y = point
        inside = False
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_at_y:
                    inside = not inside
        return inside

    def mask(self, height: int, width: int) -> np.ndarray:
        pts = np.array(
            [[int(x * width), int(y * height)] for x, y in self.polygon],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask
