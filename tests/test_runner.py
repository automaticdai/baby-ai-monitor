import numpy as np

from babymon.bus import EventBus, Subscription
from babymon.config import EventRuleConfig, RulesConfig
from babymon.events import AlertType, Frame, MotionEnergy
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
