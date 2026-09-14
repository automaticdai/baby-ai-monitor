import threading

from babymon.bus import EventBus, ReorderBuffer, Subscription
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


def test_subscription_drops_observations_when_full():
    """Detector threads must never stall behind a slow consumer.

    When a subscription's queue is full, put() silently drops observations
    and increments the dropped counter, returning without blocking.
    """
    sub = Subscription(maxsize=2)
    obs1 = motion(1.0)
    obs2 = motion(2.0)
    obs3 = motion(3.0)
    obs4 = motion(4.0)

    sub.put(obs1)
    sub.put(obs2)
    assert sub.dropped == 0

    # Queue is full; this should be dropped silently
    sub.put(obs3)
    assert sub.dropped == 1

    # put() returns immediately without blocking
    sub.put(obs4)
    assert sub.dropped == 2

    # Verify the queue contains only the first two observations
    assert sub.get(timeout=0.1) is obs1
    assert sub.get(timeout=0.1) is obs2
    assert sub.get(timeout=0.1) is None  # obs3 and obs4 were dropped


def test_bus_publish_from_multiple_threads():
    """Concurrent publishers must not deadlock or lose observations.

    Multiple threads publishing concurrently should deliver all observations
    to the subscriber without loss (below capacity) or deadlock.
    """
    bus = EventBus()
    sub = bus.subscribe(maxsize=100)
    published = []
    lock = threading.Lock()

    def publisher(thread_id: int):
        for i in range(10):
            obs = motion(float(thread_id * 100 + i))
            bus.publish(obs)
            with lock:
                published.append(obs)

    threads = [threading.Thread(target=publisher, args=(tid,)) for tid in range(3)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Drain all observations with a reasonable timeout
    received = []
    while True:
        obs = sub.get(timeout=0.1)
        if obs is None:
            break
        received.append(obs)

    # All 30 observations (10 per thread * 3 threads) should be received
    assert len(received) == 30
    # No dropped observations (we have capacity)
    assert sub.dropped == 0
    # All published observations are in received set (identity check)
    assert set(received) == set(published)
