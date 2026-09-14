import numpy as np

from babymon.bus import EventBus, Subscription
from babymon.config import EventRuleConfig, RulesConfig
from babymon.events import AlertType, Frame, Heartbeat, MotionEnergy, Severity
from babymon.rules.engine import RuleEngine
from babymon.runner import DetectorWorker, Pipeline, run_replay
from babymon.sources.base import FrameBuffer


class ExplodingDetector:
    name = "boom"

    def __init__(self):
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        raise RuntimeError("model blew up")


class CountingDetector:
    name = "counter"

    def __init__(self):
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        return [
            MotionEnergy(
                ts=frame.ts, detector=self.name, confidence=1.0, value=0.9
            )
        ]


class ListSink:
    def __init__(self):
        self.alerts = []

    def emit(self, alert, snapshot=None):
        self.alerts.append(alert)
        return len(self.alerts)

    def close(self):
        pass


def frame(seq: int) -> Frame:
    return Frame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        ts=float(seq),
        seq=seq,
        source_id="t",
    )


def test_worker_publishes_observations():
    buf, bus = FrameBuffer(), EventBus()
    sub = bus.subscribe()
    worker = DetectorWorker(CountingDetector(), buf, bus)
    buf.put(frame(0))
    assert worker.run_once() is True
    assert sub.get(timeout=0.1).value == 0.9


class SilentDetector:
    """A healthy detector with nothing to report - a person detector watching
    an empty crib, or any detector at all in a dark room."""

    name = "person"

    def __init__(self):
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        return []


def _watchdog_engine() -> RuleEngine:
    return RuleEngine(
        config=RulesConfig(
            lost_track_after_s=1e9,  # not what these tests are about
            detector_silent_after_s=15.0,
        ),
        watched_detectors=("person", "boom"),
    )


def test_a_worker_publishes_a_heartbeat_after_every_successful_frame():
    buf, bus = FrameBuffer(), EventBus()
    sub = bus.subscribe()
    worker = DetectorWorker(CountingDetector(), buf, bus)
    buf.put(frame(0))
    worker.run_once()

    assert sub.get(timeout=0.1).value == 0.9  # the observation itself
    beat = sub.get(timeout=0.1)               # then the liveness signal
    assert isinstance(beat, Heartbeat)
    assert (beat.detector, beat.ts) == ("counter", 0.0)


def test_a_detector_that_sees_nothing_is_not_reported_as_silent():
    """The watchdog must distinguish "saw nothing" from "died".

    PersonDetector publishes nothing while no person is in frame, so before
    heartbeats a healthy detector looked dead after detector_silent_after_s
    of an empty or dark scene - and raised the same alert a genuinely dead
    one would.
    """
    buf, bus = FrameBuffer(), EventBus()
    sub = bus.subscribe()
    det = SilentDetector()
    worker = DetectorWorker(det, buf, bus)
    engine = _watchdog_engine()
    engine.watched_detectors = ("person",)

    for seq in (0, 10, 20, 30, 40):  # ts 0..40, far past the 15s threshold
        buf.put(frame(seq))
        assert worker.run_once() is True
        beat = sub.get(timeout=0.1)
        assert isinstance(beat, Heartbeat)
        assert engine.handle(beat) == []  # carries no findings
        assert engine.tick(now=beat.ts) == []

    assert det.calls == 5
    assert engine.last_observation_ts["person"] == 40.0


def test_a_detector_whose_worker_has_stopped_is_reported_as_silent():
    """The contrast case, under the identical engine config: a detector that
    has failed out publishes no heartbeat, and the watchdog says so."""
    buf, bus = FrameBuffer(), EventBus()
    sub = bus.subscribe()
    worker = DetectorWorker(ExplodingDetector(), buf, bus, max_failures=1)
    engine = _watchdog_engine()
    engine.watched_detectors = ("boom",)

    engine.tick(now=0.0)
    buf.put(frame(0))
    worker.run_once()

    assert not worker.healthy  # the run loop exits; nothing more is published
    assert sub.get(timeout=0.05) is None  # no heartbeat on the failure path
    assert engine.tick(now=10.0) == []
    alerts = engine.tick(now=20.0)
    assert [a.type for a in alerts] == [AlertType.DETECTOR_SILENT]
    assert alerts[0].metadata["detector"] == "boom"


def test_a_failing_detector_does_not_raise():
    buf, bus = FrameBuffer(), EventBus()
    worker = DetectorWorker(ExplodingDetector(), buf, bus)
    buf.put(frame(0))
    assert worker.run_once() is True
    assert worker.failures == 1
    assert worker.healthy


def test_repeated_failures_mark_the_detector_unhealthy():
    buf, bus = FrameBuffer(), EventBus()
    worker = DetectorWorker(ExplodingDetector(), buf, bus, max_failures=3)
    for seq in range(3):
        buf.put(frame(seq))
        worker.run_once()
    assert worker.failures == 3
    assert not worker.healthy


def test_stride_skips_frames():
    det = CountingDetector()
    buf, bus = FrameBuffer(), EventBus()
    worker = DetectorWorker(det, buf, bus, stride=2)
    for seq in range(4):
        buf.put(frame(seq))
        worker.run_once()
    assert det.calls == 2  # seq 0 and 2


def test_replay_is_deterministic_and_reaches_the_sink():
    class StubSource:
        source_id = "t"

        def frames(self):
            for seq in range(6):
                yield frame(seq)

        def close(self):
            pass

    def make_engine():
        return RuleEngine(
            config=RulesConfig(
                awake=EventRuleConfig(
                    enter_threshold=0.5, exit_threshold=0.1,
                    min_duration_s=2.0, exit_duration_s=3.0, cooldown_s=0.0,
                ),
                lost_track_after_s=1e9,
            )
        )

    sink_a, sink_b = ListSink(), ListSink()
    a = run_replay(StubSource(), [CountingDetector()], make_engine(), [sink_a])
    b = run_replay(StubSource(), [CountingDetector()], make_engine(), [sink_b])

    assert [x.type for x in a] == [AlertType.AWAKE]
    assert [(x.type, x.started_at) for x in a] == [
        (x.type, x.started_at) for x in b
    ]
    assert sink_a.alerts == a


def _make_awake_engine() -> RuleEngine:
    return RuleEngine(
        config=RulesConfig(
            awake=EventRuleConfig(
                enter_threshold=0.5, exit_threshold=0.1,
                min_duration_s=2.0, exit_duration_s=3.0, cooldown_s=0.0,
            ),
            lost_track_after_s=1e9,
        )
    )


class _SixFrameSource:
    source_id = "t"

    def frames(self):
        for seq in range(6):
            yield frame(seq)

    def close(self):
        pass


def test_replay_continues_past_a_sink_that_raises():
    class RaisingSink:
        def __init__(self):
            self.calls = 0

        def emit(self, alert, snapshot=None):
            self.calls += 1
            raise RuntimeError("sink blew up")

        def close(self):
            pass

    det = CountingDetector()
    sink = RaisingSink()

    alerts = run_replay(_SixFrameSource(), [det], _make_awake_engine(), [sink])

    # Every frame was still processed even though the sink raised on the
    # alert produced partway through the replay.
    assert det.calls == 6
    assert [a.type for a in alerts] == [AlertType.AWAKE]
    assert sink.calls == 1


def test_replay_passes_the_frame_image_as_the_snapshot():
    class CapturingSink:
        def __init__(self):
            self.snapshots = []

        def emit(self, alert, snapshot=None):
            self.snapshots.append(snapshot)
            return len(self.snapshots)

        def close(self):
            pass

    sink = CapturingSink()
    alerts = run_replay(
        _SixFrameSource(), [CountingDetector()], _make_awake_engine(), [sink]
    )

    assert len(alerts) > 0
    assert len(sink.snapshots) == len(alerts)
    assert all(isinstance(s, np.ndarray) for s in sink.snapshots)


def test_pipeline_consume_flushes_pending_observations_on_shutdown():
    """Observations still inside the reorder window when the consume loop
    stops must be processed, not silently dropped.

    Driving this through real threads and the wall clock would be flaky (the
    window is only 0.25s by default), so instead this drives ``_consume``
    directly with a frozen clock and a stop event that lets the loop run for
    exactly one pass before stopping - deterministic, no sleeping, no
    threads.
    """

    class _StopAfterOnePass:
        def __init__(self) -> None:
            self._remaining = 1

        def is_set(self) -> bool:
            if self._remaining <= 0:
                return True
            self._remaining -= 1
            return False

    class _NoFramesSource:
        source_id = "t"

        def frames(self):
            return iter(())

        def close(self):
            pass

    engine = RuleEngine(config=RulesConfig(lost_track_after_s=1e9))
    pipeline = Pipeline(
        source=_NoFramesSource(),
        detectors=[],
        engine=engine,
        sinks=[],
        tick_interval_s=1e9,
        now=lambda: 0.0,
    )

    # Pre-load a subscription with one observation and hand it to _consume
    # in place of a fresh one, so the observation is waiting the instant the
    # loop starts - no real wait on the bus is needed.
    presub = Subscription()
    obs = MotionEnergy(ts=0.0, detector="motion", confidence=1.0, value=0.9)
    presub.put(obs)
    pipeline.bus.subscribe = lambda maxsize=1000: presub
    pipeline.stop_event = _StopAfterOnePass()

    assert "motion" not in engine.last_observation_ts
    pipeline._consume()

    # engine.handle() unconditionally records last_observation_ts before any
    # state-machine logic, so this proves the observation reached the engine
    # via flush() rather than being discarded with the reorder buffer.
    assert engine.last_observation_ts.get("motion") == 0.0


class _DyingSource:
    """Delivers a few frames, then the device goes away mid-stream."""

    source_id = "cam0"

    def __init__(self, frames_before_failure: int = 2) -> None:
        self.frames_before_failure = frames_before_failure
        self.closed = False

    def frames(self):
        for seq in range(self.frames_before_failure):
            yield frame(seq)
        raise RuntimeError("device disappeared")

    def close(self):
        self.closed = True


def _idle_engine() -> RuleEngine:
    # Nothing in these tests is about the watchdogs; only the source matters.
    return RuleEngine(
        config=RulesConfig(lost_track_after_s=1e9, detector_silent_after_s=1e9)
    )


def _pipeline(source, sink) -> Pipeline:
    return Pipeline(
        source=source,
        detectors=[],
        engine=_idle_engine(),
        sinks=[sink],
        tick_interval_s=1e9,
    )


def test_a_source_that_dies_mid_stream_emits_source_lost_and_exits_non_zero():
    """_capture's finally-block sets stop_event on ANY exit, so an
    unreadable source used to stop the monitor while the process still
    reported success - and a supervisor set to Restart=on-failure would not
    have restarted it. AlertType.SOURCE_LOST existed for exactly this case
    and was emitted nowhere."""
    sink = ListSink()
    pipeline = _pipeline(_DyingSource(), sink)

    assert pipeline.run() == 1
    assert pipeline.exit_reason == "error"
    assert isinstance(pipeline.capture_error, RuntimeError)

    assert [a.type for a in sink.alerts] == [AlertType.SOURCE_LOST]
    assert sink.alerts[0].severity == Severity.CRITICAL
    assert sink.alerts[0].metadata["source_id"] == "cam0"
    assert "device disappeared" in sink.alerts[0].metadata["error"]


def test_a_finite_source_reaching_eof_exits_zero_and_says_nothing():
    sink = ListSink()
    pipeline = _pipeline(_SixFrameSource(), sink)

    assert pipeline.run() == 0
    assert pipeline.exit_reason == "eof"
    assert sink.alerts == []


def test_a_source_raising_during_shutdown_is_not_a_failure():
    """stop() closes the source underneath the capture loop, so a raise at
    that point is the shutdown working, not the monitor dying."""

    class _RaisesImmediately:
        source_id = "cam0"

        def frames(self):
            raise RuntimeError("closed under us")
            yield  # pragma: no cover - makes this a generator

        def close(self):
            pass

    sink = ListSink()
    pipeline = _pipeline(_RaisesImmediately(), sink)
    pipeline.stop_event.set()  # shutdown already requested
    pipeline._capture()

    assert pipeline.exit_reason == "stopped"
    assert pipeline.exit_code == 0
    assert sink.alerts == []
