import cv2
import numpy as np
import pytest


def write_video(path, frames, fps=10):
    """Write a list of BGR arrays to an AVI file (MJPG is the most portable
    encoder available with opencv-python wheels)."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
    )
    assert writer.isOpened(), "could not open VideoWriter"
    for f in frames:
        writer.write(f)
    writer.release()
    return path


@pytest.fixture
def still_video(tmp_path):
    """Five identical dark frames."""
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
    return write_video(tmp_path / "still.avi", frames)
