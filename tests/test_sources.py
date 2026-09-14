import numpy as np
import pytest

from babymon.events import Frame
from babymon.sources.base import FrameBuffer
from babymon.sources.file import FileVideoSource


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
