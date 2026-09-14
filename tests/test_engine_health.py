from babymon.config import RulesConfig
from babymon.events import AlertType, MotionEnergy, PersonBox, Severity
from babymon.rules.engine import RuleEngine
from babymon.rules.zones import Zone

CRIB = Zone(
    name="crib", polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
)


def engine(**kwargs) -> RuleEngine:
    cfg = RulesConfig(
        lost_track_after_s=30.0,
        detector_silent_after_s=10.0,
        adult_min_duration_s=0.0,
        adult_hold_s=1.0,
    )
    return RuleEngine(config=cfg, crib_zone=CRIB, **kwargs)


def baby(ts):
    return PersonBox(
        ts=ts, detector="person", confidence=0.9,
        bbox=(0.45, 0.45, 0.60, 0.60), label="baby",
    )


def adult(ts):
    return PersonBox(
        ts=ts, detector="person", confidence=0.9,
        bbox=(0.0, 0.0, 0.5, 0.9), label="adult",
    )


def test_lost_track_alert_after_the_threshold():
    eng = engine()
    eng.handle(baby(0.0))
    assert eng.tick(now=20.0) == []
    alerts = eng.tick(now=31.0)
    assert [a.type for a in alerts] == [AlertType.LOST_TRACK]
    assert alerts[0].severity == Severity.CRITICAL


def test_lost_track_does_not_repeat_while_still_lost():
    eng = engine()
    eng.handle(baby(0.0))
    assert len(eng.tick(now=31.0)) == 1
    assert eng.tick(now=40.0) == []


def test_seeing_the_baby_again_clears_and_re_arms_lost_track():
    eng = engine()
    eng.handle(baby(0.0))
    eng.tick(now=31.0)
    eng.handle(baby(40.0))
    assert eng.tick(now=50.0) == []
    assert [a.type for a in eng.tick(now=75.0)] == [AlertType.LOST_TRACK]


def test_lost_track_is_not_suppressed_by_an_adult_being_present():
    eng = engine()
    eng.handle(baby(0.0))
    eng.handle(adult(1.0))
    assert eng.adult_present
    assert [a.type for a in eng.tick(now=31.0)] == [AlertType.LOST_TRACK]


def test_a_silent_detector_raises_an_alert():
    eng = engine(watched_detectors=("motion",))
    eng.handle(
        MotionEnergy(ts=0.0, detector="motion", confidence=1.0, value=0.0)
    )
    assert eng.tick(now=5.0) == []
    alerts = eng.tick(now=11.0)
    assert [a.type for a in alerts] == [AlertType.DETECTOR_SILENT]
    assert alerts[0].metadata["detector"] == "motion"


def test_a_detector_that_never_reports_is_silent_from_the_start():
    eng = engine(watched_detectors=("motion",))
    eng.tick(now=0.0)
    assert [a.type for a in eng.tick(now=11.0)] == [AlertType.DETECTOR_SILENT]


def test_a_recovered_detector_stops_alerting():
    eng = engine(watched_detectors=("motion",))
    eng.tick(now=0.0)
    eng.tick(now=11.0)
    eng.handle(
        MotionEnergy(ts=12.0, detector="motion", confidence=1.0, value=0.0)
    )
    assert eng.tick(now=13.0) == []
