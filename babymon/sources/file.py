"""Replay of a recorded video file.

Frame *spacing* always comes from the frame index and the file's FPS, never
from the wall clock, so the intervals between timestamps are identical on
every run.

The *origin* of those timestamps depends on the mode, and it has to:

* ``realtime=False`` (replay/tuning) pins ``start_ts`` to 0.0. The timeline is
  reproducible to the bit, which is what golden replay tests assert.
* ``realtime=True`` (watching a file as if it were a camera) defaults
  ``start_ts`` to the injected clock, the same ``time.monotonic`` the pipeline
  ticks the rule engine with. A zero-based origin here put the engine's
  observations ~10^5 seconds behind its own ticks, so every watchdog fired
  immediately and reported the machine's uptime as seconds-since-last-seen.

Pass ``start_ts`` explicitly to override either default.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterator

import cv2

from babymon.events import Frame


class FileVideoSource:
    def __init__(
        self,
        path: str | Path,
        source_id: str = "file",
        realtime: bool = True,
        start_ts: float | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = str(path)
        self.source_id = source_id
        self.realtime = realtime
        self.now = now
        if start_ts is None:
            # Realtime frames must share a clock domain with whoever consumes
            # them; replay must not depend on a clock at all.
            start_ts = now() if realtime else 0.0
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
        wall_start = self.now()
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    return
                ts = self.start_ts + seq * interval
                if self.realtime:
                    target = wall_start + seq * interval
                    delay = target - self.now()
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
