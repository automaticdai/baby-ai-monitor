"""Deterministic replay of a recorded video file.

Timestamps are derived from the frame index and the file's FPS, never from the
wall clock, so replaying the same file always produces the same timeline. That
property is what makes golden replay tests and threshold tuning possible.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2

from babymon.events import Frame


class FileVideoSource:
    def __init__(
        self,
        path: str | Path,
        source_id: str = "file",
        realtime: bool = True,
        start_ts: float = 0.0,
    ) -> None:
        self.path = str(path)
        self.source_id = source_id
        self.realtime = realtime
        self.start_ts = start_ts
        self._cap: cv2.VideoCapture | None = None

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video file: {self.path}")
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = 1.0 / fps
        seq = 0
        wall_start = time.monotonic()
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    return
                ts = self.start_ts + seq * interval
                if self.realtime:
                    target = wall_start + seq * interval
                    delay = target - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                yield Frame(image=image, ts=ts, seq=seq, source_id=self.source_id)
                seq += 1
        finally:
            cap.release()
            self._cap = None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
