"""In-process pub/sub.

This module is the designated seam: swapping ``EventBus`` for a ZeroMQ or
Redis implementation splits the pipeline across machines without touching
detectors or rules.
"""

from __future__ import annotations

import queue
import threading

from babymon.events import Observation


class Subscription:
    def __init__(self, maxsize: int = 1000) -> None:
        self._q: queue.Queue[Observation] = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def put(self, obs: Observation) -> None:
        try:
            self._q.put_nowait(obs)
        except queue.Full:
            # A subscriber that cannot keep up loses observations rather than
            # stalling every detector behind it.
            self.dropped += 1

    def get(self, timeout: float = 1.0) -> Observation | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()

    def subscribe(self, maxsize: int = 1000) -> Subscription:
        sub = Subscription(maxsize=maxsize)
        with self._lock:
            self._subs.append(sub)
        return sub

    def publish(self, obs: Observation) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub.put(obs)


class ReorderBuffer:
    """Holds observations briefly so they can be consumed in timestamp order.

    Detector threads finish out of order, but the rule engine's determinism
    depends on seeing observations sorted by ``ts``. Holding each observation
    for ``window_s`` before release buys that ordering for a bounded latency
    cost.
    """

    def __init__(self, window_s: float = 0.25) -> None:
        self.window_s = window_s
        self._pending: list[Observation] = []

    def add(self, obs: Observation) -> None:
        self._pending.append(obs)

    def drain(self, now: float) -> list[Observation]:
        ready = [o for o in self._pending if o.ts + self.window_s <= now]
        self._pending = [o for o in self._pending if o.ts + self.window_s > now]
        return sorted(ready, key=lambda o: o.ts)

    def flush(self) -> list[Observation]:
        """Release everything, for end-of-stream in replay."""
        ready, self._pending = sorted(self._pending, key=lambda o: o.ts), []
        return ready
