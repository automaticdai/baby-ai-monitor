"""Thread wiring and lifecycle.

A detector that raises is caught, counted, and isolated. It never takes down
the process - but it also never fails silently: once it exceeds its failure
budget it is marked unhealthy, stops producing, and the engine's watchdog
turns that silence into an alert.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable, Sequence

import numpy as np

from babymon.bus import EventBus, ReorderBuffer
from babymon.detectors.base import Detector
from babymon.events import Alert, Frame
from babymon.rules.engine import RuleEngine
from babymon.sinks.base import Sink
from babymon.sources.base import FrameBuffer, VideoSource

log = logging.getLogger(__name__)


class DetectorWorker:
    def __init__(
        self,
        detector: Detector,
        buffer: FrameBuffer,
        bus: EventBus,
        stride: int = 1,
        max_failures: int = 5,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.detector = detector
        self.buffer = buffer
        self.bus = bus
        self.stride = max(1, stride)
        self.max_failures = max_failures
        self.stop_event = stop_event or threading.Event()
        self.failures = 0
        self.healthy = True
        self._last_seq: int | None = None

    def run_once(self, timeout: float = 0.5) -> bool:
        """Process at most one frame. Returns True if a frame was consumed."""
        frame = self.buffer.get_latest(self._last_seq, timeout=timeout)
        if frame is None:
            return False
        self._last_seq = frame.seq
        if frame.seq % self.stride != 0:
            return True
        try:
            for obs in self.detector.process(frame):
                self.bus.publish(obs)
            self.failures = 0
        except Exception:
            self.failures += 1
            log.exception(
                "detector %s failed (%d/%d)",
                self.detector.name, self.failures, self.max_failures,
            )
            if self.failures >= self.max_failures:
                self.healthy = False
                log.error(
                    "detector %s disabled after repeated failures",
                    self.detector.name,
                )
        return True

    def run(self) -> None:
        while not self.stop_event.is_set() and self.healthy:
            self.run_once()


class Pipeline:
    def __init__(
        self,
        source: VideoSource,
        detectors: Sequence[Detector],
        engine: RuleEngine,
        sinks: Sequence[Sink],
        strides: dict[str, int] | None = None,
        tick_interval_s: float = 1.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.engine = engine
        self.sinks = list(sinks)
        self.now = now
        self.tick_interval_s = tick_interval_s
        self.buffer = FrameBuffer()
        self.bus = EventBus()
        self.stop_event = threading.Event()
        strides = strides or {}
        self.workers = [
            DetectorWorker(
                d, self.buffer, self.bus,
                stride=strides.get(d.name, 1),
                stop_event=self.stop_event,
            )
            for d in detectors
        ]
        self._latest_image: np.ndarray | None = None
        self._threads: list[threading.Thread] = []

    def _capture(self) -> None:
        try:
            for frame in self.source.frames():
                if self.stop_event.is_set():
                    return
                self._latest_image = frame.image
                self.buffer.put(frame)
        finally:
            self.stop_event.set()

    def _consume(self) -> None:
        sub = self.bus.subscribe()
        reorder = ReorderBuffer(self.engine.config.reorder_window_s)
        next_tick = self.now()
        while not self.stop_event.is_set():
            obs = sub.get(timeout=0.2)
            if obs is not None:
                reorder.add(obs)
            now = self.now()
            for ready in reorder.drain(now):
                self._dispatch(self.engine.handle(ready))
            if now >= next_tick:
                self._dispatch(self.engine.tick(now))
                next_tick = now + self.tick_interval_s

    def _dispatch(self, alerts: Iterable[Alert]) -> None:
        for alert in alerts:
            log.info("ALERT %s at %.1f", alert.type.value, alert.started_at)
            for sink in self.sinks:
                try:
                    sink.emit(alert, self._latest_image)
                except Exception:
                    log.exception("sink %s failed", type(sink).__name__)

    def run(self) -> None:
        self._threads = [
            threading.Thread(target=self._capture, name="capture", daemon=True),
            threading.Thread(target=self._consume, name="rules", daemon=True),
            *[
                threading.Thread(target=w.run, name=w.detector.name, daemon=True)
                for w in self.workers
            ],
        ]
        for t in self._threads:
            t.start()
        try:
            while not self.stop_event.wait(timeout=0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()
        self.source.close()
        for t in self._threads:
            t.join(timeout=2.0)
        for sink in self.sinks:
            sink.close()


def run_replay(
    source: VideoSource,
    detectors: Sequence[Detector],
    engine: RuleEngine,
    sinks: Sequence[Sink],
) -> list[Alert]:
    """Process a source synchronously, in frame order.

    No threads and no wall clock, so the same input always produces the same
    alert timeline. This is what golden replay tests and threshold tuning run
    against.
    """
    alerts: list[Alert] = []
    for frame in source.frames():
        observations = []
        for detector in detectors:
            try:
                observations.extend(detector.process(frame))
            except Exception:
                log.exception("detector %s failed during replay", detector.name)
        for obs in sorted(observations, key=lambda o: o.ts):
            alerts.extend(engine.handle(obs))
        alerts.extend(engine.tick(frame.ts))
    for alert in alerts:
        for sink in sinks:
            sink.emit(alert, None)
    return alerts
