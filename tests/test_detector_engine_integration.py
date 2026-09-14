"""The detector/engine seam, driven end to end.

Every other test in the suite exercises one side of this boundary: the
detector tests assert labels, the engine tests feed hand-written labels in.
Nothing checked that the labels a real ``PersonDetector`` actually produces
mean what the ``RuleEngine`` assumes they mean - which is how a baby leaving
the crib came to be labelled "adult", activating suppression and silencing the
monitor at the exact moment something happened.

So these tests use a real detector and a real engine, with only the model
faked, and assert on the alert stream and the engine state that results.
"""

import numpy as np

from babymon.config import RulesConfig
from babymon.detectors.person import PersonDetector
from babymon.events import AlertType, Frame, MotionEnergy, Severity
from babymon.rules.engine import RuleEngine
from babymon.rules.zones import Zone

CRIB = Zone(
    name="crib", polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
)


class FakeModel:
    """Stands in for YOLO. ``detections`` is rewritten between frames so a
    single detector instance can watch something move across the scene."""

    def __init__(self, detections=()):
        self.detections = list(detections)

    def detect_persons(self, image):
        return list(self.detections)


def frame(ts: float) -> Frame:
    return Frame(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        ts=ts,
        seq=int(ts * 10),
        source_id="t",
    )


def small_box_at(cx: float, cy: float = 0.5):
    """A baby-sized box (area 0.01, far below baby_max_area) centred at cx."""
    return ((cx - 0.05, cy - 0.05, cx + 0.05, cy + 0.05), 0.9)


def build() -> tuple[FakeModel, PersonDetector, RuleEngine]:
    model = FakeModel()
    detector = PersonDetector(model=model, crib_zone=CRIB)
    engine = RuleEngine(
        config=RulesConfig(
            lost_track_after_s=30.0,
            detector_silent_after_s=1e9,
            adult_min_duration_s=0.0,
            adult_hold_s=5.0,
        ),
        crib_zone=CRIB,
    )
    return model, detector, engine


def step(detector, engine, model, ts, detections) -> list:
    """One frame through the real detector into the real engine."""
    model.detections = detections
    alerts = []
    for obs in detector.process(frame(ts)):
        alerts.extend(engine.handle(obs))
    return alerts


def test_a_baby_walking_out_of_the_crib_does_not_silence_the_monitor():
    model, detector, engine = build()

    # Three frames with the baby inside the crib: labelled "baby", so
    # last_baby_seen advances and nothing is suppressed.
    inside_alerts = []
    for ts, cx in [(0.0, 0.50), (1.0, 0.60), (2.0, 0.70)]:
        inside_alerts += step(detector, engine, model, ts, [small_box_at(cx)])
    assert inside_alerts == []
    assert engine.last_baby_seen == 2.0
    assert not engine.adult_present

    # Same small box, now outside the crib polygon. The label is "unknown":
    # not "adult" (which would activate suppression and mute every other
    # alert) and not "baby" (which the detector has no way to establish).
    labels = []
    outside_alerts = []
    for ts, cx in [(3.0, 0.85), (4.0, 0.90), (5.0, 0.95)]:
        model.detections = [small_box_at(cx)]
        observations = detector.process(frame(ts))
        labels += [o.label for o in observations]
        for obs in observations:
            outside_alerts.extend(engine.handle(obs))

    assert labels == ["unknown", "unknown", "unknown"]
    assert outside_alerts == []
    # The heart of it: a departing baby must NOT look like an adult.
    assert not engine.adult_present
    # ...and must not keep refreshing the lost-track clock either.
    assert engine.last_baby_seen == 2.0

    # Because suppression never engaged, an ordinary alert still gets out.
    awake = []
    for ts, value in [(6.0, 0.3), (7.0, 0.3), (8.5, 0.3)]:
        awake.extend(
            engine.handle(
                MotionEnergy(
                    ts=ts, detector="motion", confidence=1.0, value=value
                )
            )
        )
    assert [a.type for a in awake] == [AlertType.AWAKE]

    # And the watchdog turns the disappearance into noise, at full severity
    # because no adult is present to account for it.
    lost = engine.tick(now=40.0)
    assert [a.type for a in lost] == [AlertType.LOST_TRACK]
    assert lost[0].severity == Severity.CRITICAL
    assert lost[0].metadata["seconds_since_last_seen"] == 38.0


def test_no_zone_exit_is_reported_for_a_departing_baby():
    """ZONE_EXIT is not implemented and must not be claimed.

    The engine's zone-exit state machine only sees observations labelled
    "baby", and the detector only applies that label inside the crib, so the
    real pipeline cannot produce one. This pins that honestly rather than
    leaving a half-working alert in place.
    """
    model, detector, engine = build()
    alerts = []
    for i, cx in enumerate([0.50, 0.70, 0.85, 0.90, 0.95, 0.95, 0.95]):
        alerts += step(detector, engine, model, float(i), [small_box_at(cx)])
    assert [a.type for a in alerts] == []


def test_an_adult_leaning_over_the_crib_still_suppresses():
    """The contrast case: relabelling must not have cost us suppression.

    A large box is an adult wherever its centre falls - including inside the
    crib polygon, which is exactly what leaning over the cot looks like.
    """
    model, detector, engine = build()
    big = ((0.05, 0.05, 0.95, 0.95), 0.9)  # area 0.81, well over baby_max_area

    labels = []
    alerts = []
    for ts, value in [(0.0, 0.3), (1.0, 0.3), (2.5, 0.3)]:
        model.detections = [big]
        observations = detector.process(frame(ts))
        labels += [o.label for o in observations]
        for obs in observations:
            alerts.extend(engine.handle(obs))
        alerts.extend(
            engine.handle(
                MotionEnergy(
                    ts=ts, detector="motion", confidence=1.0, value=value
                )
            )
        )

    assert labels == ["adult", "adult", "adult"]
    assert engine.adult_present
    assert alerts == []  # AWAKE suppressed by the caregiver in frame
