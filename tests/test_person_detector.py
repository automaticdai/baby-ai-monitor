import numpy as np

from babymon.detectors.person import PersonDetector
from babymon.events import Frame
from babymon.rules.zones import Zone

CRIB = Zone(
    name="crib", polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
)


class FakeModel:
    """Stands in for YOLO so detector logic is tested without a model file."""

    def __init__(self, detections):
        self.detections = detections

    def detect_persons(self, image):
        return self.detections


def frame() -> Frame:
    return Frame(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        ts=1.0,
        seq=0,
        source_id="t",
    )


def detector(detections, **kwargs) -> PersonDetector:
    return PersonDetector(model=FakeModel(detections), crib_zone=CRIB, **kwargs)


def test_small_person_inside_the_crib_is_labelled_baby():
    det = detector([((0.45, 0.45, 0.60, 0.60), 0.9)])
    (obs,) = det.process(frame())
    assert obs.label == "baby"
    assert obs.detector == "person"


def test_large_person_inside_the_crib_is_labelled_adult():
    # Centre is inside the crib polygon but the box is far too big for a baby.
    det = detector([((0.05, 0.05, 0.95, 0.95), 0.9)])
    (obs,) = det.process(frame())
    assert obs.label == "adult"


def test_small_person_outside_the_crib_zone_is_unknown():
    # Small and outside the crib is genuinely ambiguous - the baby who
    # climbed out, a pet, or an adult far enough away to fall under the area
    # threshold all look like this. Labelling it "adult" (which this detector
    # used to do) would activate suppression and silence the monitor exactly
    # when a baby left the crib; "unknown" neither suppresses nor refreshes
    # last_baby_seen, so the lost-track watchdog still fires.
    det = detector([((0.02, 0.02, 0.12, 0.12), 0.9)])
    (obs,) = det.process(frame())
    assert obs.label == "unknown"


def test_large_person_outside_the_crib_zone_is_still_adult():
    # Size is checked before position: a big box is an adult wherever it
    # stands. Centre (0.45, 0.15) is above the crib polygon's top edge (y=0.2)
    # and the area is 0.27, over baby_max_area.
    det = detector([((0.0, 0.0, 0.9, 0.3), 0.9)])
    (obs,) = det.process(frame())
    assert obs.center == (0.45, 0.15)
    assert not CRIB.contains(obs.center)
    assert obs.label == "adult"


def test_low_confidence_detections_are_dropped():
    det = detector([((0.45, 0.45, 0.60, 0.60), 0.2)], conf_threshold=0.4)
    assert det.process(frame()) == []


def test_without_a_crib_zone_people_are_unknown(caplog):
    det = PersonDetector(
        model=FakeModel([((0.45, 0.45, 0.60, 0.60), 0.9)]), crib_zone=None
    )
    (obs,) = det.process(frame())
    assert obs.label == "unknown"
    assert "Baby/adult attribution disabled" in caplog.text


def test_multiple_detections_are_all_reported():
    det = detector(
        [((0.45, 0.45, 0.60, 0.60), 0.9), ((0.01, 0.01, 0.9, 0.99), 0.8)]
    )
    labels = sorted(o.label for o in det.process(frame()))
    assert labels == ["adult", "baby"]


def test_person_at_exact_baby_max_area_boundary_is_baby():
    # Box with area exactly equal to baby_max_area (0.25) and center inside crib
    # is labeled baby. Dimensions: 0.5 x 0.5 = 0.25
    det = detector([((0.25, 0.25, 0.75, 0.75), 0.9)])
    (obs,) = det.process(frame())
    assert obs.label == "baby"


def test_person_above_baby_max_area_boundary_is_adult():
    # Box with area above baby_max_area (0.26 > 0.25) and center inside crib
    # is labeled adult. Dimensions: 0.52 x 0.5 = 0.26
    det = detector([((0.24, 0.25, 0.76, 0.75), 0.9)])
    (obs,) = det.process(frame())
    assert obs.label == "adult"
