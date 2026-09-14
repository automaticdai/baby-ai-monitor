import numpy as np
import pytest

from babymon.events import (
    Alert,
    AlertType,
    Frame,
    Heartbeat,
    MotionEnergy,
    Observation,
    PersonBox,
    Severity,
)


def test_frame_carries_timing_metadata():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    frame = Frame(image=img, ts=12.5, seq=3, source_id="cam0")
    assert (frame.ts, frame.seq, frame.source_id) == (12.5, 3, "cam0")


def test_person_box_derives_normalised_centre_and_area():
    box = PersonBox(
        ts=1.0, detector="person", confidence=0.9, bbox=(0.2, 0.2, 0.6, 0.4)
    )
    assert box.center == pytest.approx((0.4, 0.3))
    assert box.area == pytest.approx(0.08)


def test_person_box_label_defaults_to_unknown():
    box = PersonBox(ts=1.0, detector="person", confidence=0.9, bbox=(0, 0, 1, 1))
    assert box.label == "unknown"


def test_motion_energy_is_an_observation_with_a_value():
    obs = MotionEnergy(ts=2.0, detector="motion", confidence=1.0, value=0.42)
    assert obs.value == 0.42
    assert obs.ts == 2.0


def test_alert_defaults_to_an_open_ended_episode():
    alert = Alert(
        type=AlertType.AWAKE, severity=Severity.INFO, started_at=5.0
    )
    assert alert.ended_at is None
    assert alert.metadata == {}
    assert alert.confidence == 1.0


def test_heartbeat_is_an_observation_carrying_only_liveness():
    beat = Heartbeat(ts=3.0, detector="person")
    assert isinstance(beat, Observation)
    assert (beat.detector, beat.ts, beat.confidence) == ("person", 3.0, 1.0)
    # No findings of any kind: a heartbeat says the detector ran, nothing
    # about what it saw. Anything else would make it a second, silent channel
    # for detection results.
    assert not isinstance(beat, (PersonBox, MotionEnergy))
    assert set(vars(beat)) == {"ts", "detector", "confidence"}
