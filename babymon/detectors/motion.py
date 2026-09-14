"""Frame-differencing motion energy, optionally restricted to a zone."""

from __future__ import annotations

import cv2
import numpy as np

from babymon.events import Frame, MotionEnergy, Observation
from babymon.rules.zones import Zone


class MotionDetector:
    name = "motion"

    def __init__(
        self,
        zone: Zone | None = None,
        pixel_threshold: int = 25,
        blur_ksize: int = 5,
    ) -> None:
        self.zone = zone
        self.pixel_threshold = pixel_threshold
        self.blur_ksize = blur_ksize
        self._prev: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._mask_shape: tuple[int, int] | None = None

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

    def _zone_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        if self.zone is None:
            return None
        if self._mask is None or self._mask_shape != shape:
            self._mask = self.zone.mask(shape[0], shape[1])
            self._mask_shape = shape
        return self._mask

    def process(self, frame: Frame) -> list[Observation]:
        current = self._prepare(frame.image)
        prev, self._prev = self._prev, current
        if prev is None or prev.shape != current.shape:
            return []

        diff = cv2.absdiff(prev, current)
        _, changed = cv2.threshold(
            diff, self.pixel_threshold, 255, cv2.THRESH_BINARY
        )

        mask = self._zone_mask(current.shape[:2])
        if mask is not None:
            changed = cv2.bitwise_and(changed, changed, mask=mask)
            total = int(np.count_nonzero(mask))
        else:
            total = changed.size

        value = 0.0 if total == 0 else float(np.count_nonzero(changed)) / total
        return [
            MotionEnergy(
                ts=frame.ts, detector=self.name, confidence=1.0, value=value
            )
        ]
