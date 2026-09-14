"""Live webcam capture with reconnect-on-failure."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterator

import cv2

from babymon.events import Frame

log = logging.getLogger(__name__)


class WebcamSource:
    def __init__(
        self,
        device: int = 0,
        source_id: str = "cam0",
        now: Callable[[], float] = time.monotonic,
        max_backoff_s: float = 30.0,
    ) -> None:
        self.device = device
        self.source_id = source_id
        self.now = now
        self.max_backoff_s = max_backoff_s
        self._cap: cv2.VideoCapture | None = None
        self._stopped = False
        self.consecutive_failures = 0

    def frames(self) -> Iterator[Frame]:
        seq = 0
        backoff = 1.0
        while not self._stopped:
            cap = cv2.VideoCapture(self.device)
            if not cap.isOpened():
                self.consecutive_failures += 1
                log.warning(
                    "camera %s unavailable, retrying in %.1fs",
                    self.device, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                continue
            self._cap = cap
            backoff = 1.0
            self.consecutive_failures = 0
            while not self._stopped:
                ok, image = cap.read()
                if not ok:
                    log.warning("camera %s read failed, reconnecting", self.device)
                    self.consecutive_failures += 1
                    break
                yield Frame(
                    image=image, ts=self.now(), seq=seq, source_id=self.source_id
                )
                seq += 1
            cap.release()
            self._cap = None

    def close(self) -> None:
        self._stopped = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None
