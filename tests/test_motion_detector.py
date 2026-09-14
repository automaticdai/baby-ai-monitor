import numpy as np

from babymon.detectors.motion import MotionDetector
from babymon.events import Frame
from babymon.rules.zones import Zone

CRIB = Zone(name="crib", polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])


def frame(seq: int, image: np.ndarray) -> Frame:
    return Frame(image=image, ts=float(seq), seq=seq, source_id="t")


def blank() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_first_frame_produces_no_observation():
    assert MotionDetector().process(frame(0, blank())) == []


def test_identical_frames_produce_near_zero_energy():
    det = MotionDetector()
    det.process(frame(0, blank()))
    (obs,) = det.process(frame(1, blank()))
    assert obs.value == 0.0
    assert obs.detector == "motion"
    assert obs.ts == 1.0


def test_changed_pixels_raise_motion_energy():
    det = MotionDetector()
    det.process(frame(0, blank()))
    moved = blank()
    moved[10:60, 10:60] = 255  # a quarter of the image
    (obs,) = det.process(frame(1, moved))
    assert obs.value > 0.2


def test_motion_outside_the_zone_is_ignored():
    det = MotionDetector(zone=CRIB)  # zone is the left half
    det.process(frame(0, blank()))
    moved = blank()
    moved[10:90, 60:95] = 255  # entirely in the right half
    (obs,) = det.process(frame(1, moved))
    assert obs.value == 0.0


def test_motion_inside_the_zone_is_counted():
    det = MotionDetector(zone=CRIB)
    det.process(frame(0, blank()))
    moved = blank()
    moved[10:90, 5:45] = 255  # left half, inside the zone
    (obs,) = det.process(frame(1, moved))
    assert obs.value > 0.2
