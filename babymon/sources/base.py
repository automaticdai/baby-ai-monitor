"""Video source protocol and the latest-wins frame buffer."""

from __future__ import annotations

import threading
from typing import Iterator, Protocol

from babymon.events import Frame


class VideoSource(Protocol):
    source_id: str

    def frames(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class FrameBuffer:
    """Holds only the newest frame.

    Deliberately not a queue: for real-time alerting a stale frame has no
    value, so under load we drop old frames rather than accumulate a backlog
    that pushes every alert further behind live. ``drops`` is a health metric.
    """

    def __init__(self) -> None:
        self._frame: Frame | None = None
        self._consumed = True
        self._drops = 0
        self._cond = threading.Condition()

    @property
    def drops(self) -> int:
        with self._cond:
            return self._drops

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None and not self._consumed:
                self._drops += 1
            self._frame = frame
            self._consumed = False
            self._cond.notify_all()

    def get_latest(
        self, last_seq: int | None = None, timeout: float = 1.0
    ) -> Frame | None:
        with self._cond:
            def ready() -> bool:
                return self._frame is not None and (
                    last_seq is None or self._frame.seq != last_seq
                )

            if not ready():
                self._cond.wait_for(ready, timeout=timeout)
            if not ready():
                return None
            self._consumed = True
            return self._frame
