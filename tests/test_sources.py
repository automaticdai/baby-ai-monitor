import time

import numpy as np
import pytest

from babymon.events import Frame
from babymon.sources.base import FrameBuffer
from babymon.sources.file import FileVideoSource
from babymon.sources.webcam import WebcamSource


def make_frame(seq: int) -> Frame:
    return Frame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        ts=float(seq),
        seq=seq,
        source_id="t",
    )


def test_buffer_returns_the_latest_frame():
    buf = FrameBuffer()
    buf.put(make_frame(1))
    assert buf.get_latest(timeout=0.1).seq == 1


def test_buffer_overwrites_unconsumed_frames_and_counts_drops():
    buf = FrameBuffer()
    buf.put(make_frame(1))
    buf.put(make_frame(2))
    assert buf.drops == 1
    assert buf.get_latest(timeout=0.1).seq == 2


def test_buffer_returns_none_when_no_newer_frame_arrives():
    buf = FrameBuffer()
    buf.put(make_frame(1))
    assert buf.get_latest(timeout=0.1).seq == 1
    assert buf.get_latest(last_seq=1, timeout=0.05) is None


def test_file_source_yields_sequential_frames(still_video):
    src = FileVideoSource(still_video, realtime=False)
    frames = list(src.frames())
    assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
    assert frames[0].image.shape == (240, 320, 3)


def test_file_source_timestamps_are_deterministic(still_video):
    src = FileVideoSource(still_video, realtime=False, start_ts=100.0)
    frames = list(src.frames())
    # 10 fps fixture -> 0.1s apart, derived from frame index not wall clock.
    assert frames[0].ts == pytest.approx(100.0)
    assert frames[2].ts == pytest.approx(100.2)


def test_replaying_the_same_file_gives_identical_timestamps(still_video):
    a = [f.ts for f in FileVideoSource(still_video, realtime=False).frames()]
    b = [f.ts for f in FileVideoSource(still_video, realtime=False).frames()]
    assert a == b


def test_realtime_file_source_defaults_to_the_monotonic_clock(still_video):
    """The clock-domain bug: `--source clip.mp4` without `--replay`.

    Pipeline ticks the rule engine with time.monotonic (~10^5 on any machine
    that has been up a while). A file source stamping frames from 0.0 put
    every observation ~10^5 seconds in the engine's past, so the lost-track
    watchdog fired on the first tick and reported the machine's uptime as
    seconds-since-last-seen.
    """
    before = time.monotonic()
    src = FileVideoSource(still_video, realtime=True)
    after = time.monotonic()
    assert before <= src.start_ts <= after

    gen = src.frames()
    try:
        first = next(gen)
    finally:
        gen.close()
        src.close()
    assert first.ts == pytest.approx(src.start_ts)


def test_realtime_file_source_uses_the_injected_clock(still_video):
    src = FileVideoSource(still_video, realtime=True, now=lambda: 12345.0)
    assert src.start_ts == 12345.0

    gen = src.frames()
    try:
        # Only the first frame: with a frozen clock later frames would sleep
        # out their real inter-frame interval.
        first = next(gen)
    finally:
        gen.close()
        src.close()
    assert first.ts == pytest.approx(12345.0)


def test_replay_file_source_ignores_the_clock_and_starts_at_zero(still_video):
    """Replay determinism must not depend on a clock at all, which is what
    keeps the golden timeline reproducible."""
    src = FileVideoSource(still_video, realtime=False, now=lambda: 12345.0)
    assert src.start_ts == 0.0
    assert [f.ts for f in src.frames()][0] == 0.0


def test_an_explicit_start_ts_overrides_both_defaults(still_video):
    realtime = FileVideoSource(
        still_video, realtime=True, start_ts=7.0, now=lambda: 12345.0
    )
    replay = FileVideoSource(still_video, realtime=False, start_ts=7.0)
    assert (realtime.start_ts, replay.start_ts) == (7.0, 7.0)


class _AlwaysFailsToOpen:
    """Fake cv2.VideoCapture: the device never opens."""

    def isOpened(self):
        return False

    def read(self):
        return False, None

    def release(self):
        pass


class _OpensButAlwaysFailsToRead:
    """Fake cv2.VideoCapture: open succeeds, every read() fails."""

    def __init__(self):
        self.read_calls = 0

    def isOpened(self):
        return True

    def read(self):
        self.read_calls += 1
        if self.read_calls > 200:
            # Safety net: a regression that never applies backoff on the
            # read-failure path would otherwise busy-loop forever here.
            raise AssertionError("read() called too many times; runaway loop")
        return False, None

    def release(self):
        pass


class _FlappingCapture:
    """Fake cv2.VideoCapture: isOpened() always succeeds, read() follows a
    fixed pass/fail script (True = deliver a frame, False = fail)."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def isOpened(self):
        return True

    def read(self):
        ok = self.script[self.calls]
        self.calls += 1
        if ok:
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        return False, None

    def release(self):
        pass


def test_webcam_source_read_failures_grow_failures_and_backoff():
    """Reproduces the read-failure busy-loop bug: without backoff on this
    path, sleep() is never called and the loop spins with zero delay."""
    sleeps: list[float] = []
    src = None

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 5:
            src.close()

    src = WebcamSource(
        capture_factory=lambda device: _OpensButAlwaysFailsToRead(),
        sleep=fake_sleep,
        now=lambda: 0.0,
    )
    frames = list(src.frames())

    assert frames == []
    assert src.consecutive_failures >= 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_webcam_source_open_failures_grow_backoff_capped_at_max():
    sleeps: list[float] = []
    src = None

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 8:
            src.close()

    src = WebcamSource(
        capture_factory=lambda device: _AlwaysFailsToOpen(),
        sleep=fake_sleep,
        now=lambda: 0.0,
        max_backoff_s=10.0,
    )
    frames = list(src.frames())

    assert frames == []
    assert src.consecutive_failures >= 8
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]


def test_webcam_source_resets_failures_and_backoff_after_a_successful_frame():
    # fail x3, succeed, fail x2, succeed
    script = [False, False, False, True, False, False, True]
    fake_cap = _FlappingCapture(script)
    sleeps: list[float] = []

    src = WebcamSource(
        capture_factory=lambda device: fake_cap,
        sleep=lambda seconds: sleeps.append(seconds),
        now=lambda: 0.0,
    )
    gen = src.frames()

    next(gen)  # succeeds after 3 read failures
    assert sleeps == [1.0, 2.0, 4.0]
    assert src.consecutive_failures == 0

    next(gen)  # succeeds again after 2 more read failures; backoff restarted
    assert sleeps == [1.0, 2.0, 4.0, 1.0, 2.0]
    assert src.consecutive_failures == 0

    src.close()
