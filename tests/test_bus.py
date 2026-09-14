from babymon.bus import EventBus, ReorderBuffer
from babymon.events import MotionEnergy


def motion(ts: float, value: float = 0.5) -> MotionEnergy:
    return MotionEnergy(ts=ts, detector="motion", confidence=1.0, value=value)


def test_subscriber_receives_published_observations():
    bus = EventBus()
    sub = bus.subscribe()
    obs = motion(1.0)
    bus.publish(obs)
    assert sub.get(timeout=0.1) is obs


def test_every_subscriber_gets_its_own_copy():
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    obs = motion(1.0)
    bus.publish(obs)
    assert a.get(timeout=0.1) is obs
    assert b.get(timeout=0.1) is obs


def test_get_returns_none_when_nothing_published():
    assert EventBus().subscribe().get(timeout=0.01) is None


def test_reorder_buffer_holds_observations_inside_the_window():
    buf = ReorderBuffer(window_s=0.5)
    buf.add(motion(2.0))
    assert buf.drain(now=2.1) == []


def test_reorder_buffer_releases_in_timestamp_order():
    buf = ReorderBuffer(window_s=0.5)
    buf.add(motion(2.0))
    buf.add(motion(1.0))
    released = buf.drain(now=2.6)
    assert [o.ts for o in released] == [1.0, 2.0]


def test_reorder_buffer_does_not_re_release():
    buf = ReorderBuffer(window_s=0.5)
    buf.add(motion(1.0))
    assert len(buf.drain(now=2.0)) == 1
    assert buf.drain(now=3.0) == []
