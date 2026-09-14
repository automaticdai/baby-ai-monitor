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
from babymon.events import Alert, AlertType, Frame, Heartbeat
from babymon.rules.engine import SEVERITY, RuleEngine
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
            # Published whatever process() returned, including nothing. The
            # watchdog needs "this detector is still running" to be separable
            # from "this detector had something to say" - otherwise a person
            # detector watching an empty crib looks exactly like a dead one.
            self.bus.publish(
                Heartbeat(ts=frame.ts, detector=self.detector.name)
            )
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
        #: Why capture ended. "running" until it does; then "eof" (a finite
        #: source ran out), "stopped" (shutdown was asked for) or "error".
        self.exit_reason = "running"
        self.capture_error: BaseException | None = None

    def _capture(self) -> None:
        """Capture frames, and record WHY the loop ended.

        The finally-block below sets stop_event on any exit at all, including
        an exception, so without this distinction an unreadable source stopped
        the monitor and the process still reported success - a supervisor set
        to Restart=on-failure would not have restarted it.
        """
        try:
            for frame in self.source.frames():
                if self.stop_event.is_set():
                    self.exit_reason = "stopped"
                    return
                self._latest_image = frame.image
                self.buffer.put(frame)
            self.exit_reason = "stopped" if self.stop_event.is_set() else "eof"
        except Exception as exc:
            if self.stop_event.is_set():
                # stop() closes the source underneath this loop, so a raise
                # here during a deliberate shutdown is expected, not a fault.
                self.exit_reason = "stopped"
                log.debug("source raised during shutdown", exc_info=True)
            else:
                self.exit_reason = "error"
                self.capture_error = exc
                log.exception("capture failed; the monitor is now blind")
                self._dispatch([self._source_lost_alert(exc)])
        finally:
            self.stop_event.set()

    def _source_lost_alert(self, exc: BaseException) -> Alert:
        """The monitor can no longer see. AlertType.SOURCE_LOST exists for
        exactly this, and had never been emitted anywhere."""
        return Alert(
            type=AlertType.SOURCE_LOST,
            severity=SEVERITY[AlertType.SOURCE_LOST],
            started_at=self.now(),
            confidence=1.0,
            metadata={
                "source_id": getattr(self.source, "source_id", "unknown"),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    @property
    def exit_code(self) -> int:
        """0 only when the monitor stopped for a reason someone chose.

        A finite source reaching EOF is a clean finish, and so is an
        explicit stop or Ctrl-C. Anything else means the monitor died and
        must be restartable by a supervisor.
        """
        return 0 if self.exit_reason in ("eof", "stopped") else 1

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
        # A live stream always has something pending inside the reorder
        # window at stop time; flush it through rather than dropping it.
        for ready in reorder.flush():
            self._dispatch(self.engine.handle(ready))

    def _dispatch(self, alerts: Iterable[Alert]) -> None:
        for alert in alerts:
            log.info("ALERT %s at %.1f", alert.type.value, alert.started_at)
            for sink in self.sinks:
                try:
                    sink.emit(alert, self._latest_image)
                except Exception:
                    log.exception("sink %s failed", type(sink).__name__)

    def run(self) -> int:
        """Run until the source ends or something stops us. Returns
        ``exit_code`` so a caller can distinguish a finish from a failure."""
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
            self.exit_reason = "stopped"
        finally:
            self.stop()
        return self.exit_code

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
            else:
                # Same liveness contract as DetectorWorker. Replay is the
                # tuning path, so it has to reproduce what the live pipeline
                # would have done - including not reporting a quiet detector
                # as a dead one.
                observations.append(
                    Heartbeat(ts=frame.ts, detector=detector.name)
                )
        frame_alerts: list[Alert] = []
        for obs in sorted(observations, key=lambda o: o.ts):
            frame_alerts.extend(engine.handle(obs))
        frame_alerts.extend(engine.tick(frame.ts))
        # Dispatch as each frame's alerts are produced, not after the whole
        # replay finishes: that is what lets each alert carry its own
        # frame's image as the snapshot, and what keeps one bad sink from
        # losing the rest of an already-computed timeline.
        for alert in frame_alerts:
            for sink in sinks:
                try:
                    sink.emit(alert, frame.image)
                except Exception:
                    log.exception(
                        "sink %s failed during replay", type(sink).__name__
                    )
        alerts.extend(frame_alerts)
    return alerts
