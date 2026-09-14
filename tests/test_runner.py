import numpy as np

from babymon.bus import EventBus
from babymon.config import EventRuleConfig, RulesConfig
from babymon.events import AlertType, Frame, MotionEnergy
from babymon.rules.engine import RuleEngine
from babymon.runner import DetectorWorker, run_replay
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
