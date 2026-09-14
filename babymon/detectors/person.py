"""Person detection plus baby/adult attribution.

Attribution is a geometric heuristic, not a model: nothing off the shelf
classifies "infant". A person box whose centre lies inside the crib zone and
whose area is below a threshold is the baby; anything else is an adult.

This holds for a fixed camera over a crib and breaks if the camera moves,
which is why the rule engine watches for large shifts in scene geometry and
warns rather than silently mis-attributing every later detection.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from babymon.events import BBox, Frame, Observation, PersonBox
from babymon.rules.zones import Zone


class PersonModel(Protocol):
    def detect_persons(self, image: np.ndarray) -> list[tuple[BBox, float]]: ...


class YoloPersonModel:
    """Ultralytics YOLO adapter. Returns normalised boxes for class 0 only."""

    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self._model = YOLO(model_path)

    def detect_persons(self, image: np.ndarray) -> list[tuple[BBox, float]]:
        height, width = image.shape[:2]
        results = self._model.predict(
            image, classes=[0], verbose=False
        )  # class 0 == person
        out: list[tuple[BBox, float]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                out.append(
                    (
                        (x1 / width, y1 / height, x2 / width, y2 / height),
                        float(box.conf[0]),
                    )
                )
        return out


class PersonDetector:
    name = "person"

    def __init__(
        self,
        model: PersonModel,
        crib_zone: Zone | None = None,
        conf_threshold: float = 0.4,
        baby_max_area: float = 0.25,
    ) -> None:
        self.model = model
        self.crib_zone = crib_zone
        self.conf_threshold = conf_threshold
        self.baby_max_area = baby_max_area

    def _label(self, box: PersonBox) -> str:
        if self.crib_zone is None:
            return "adult"
        in_crib = self.crib_zone.contains(box.center)
        return "baby" if in_crib and box.area <= self.baby_max_area else "adult"

    def process(self, frame: Frame) -> list[Observation]:
        out: list[Observation] = []
        for bbox, conf in self.model.detect_persons(frame.image):
            if conf < self.conf_threshold:
                continue
            box = PersonBox(
                ts=frame.ts, detector=self.name, confidence=conf, bbox=bbox
            )
            out.append(
                PersonBox(
                    ts=box.ts,
                    detector=box.detector,
                    confidence=box.confidence,
                    bbox=box.bbox,
                    label=self._label(box),
                )
            )
        return out
