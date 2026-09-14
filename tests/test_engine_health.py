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


def motion(ts, value):
    return MotionEnergy(ts=ts, detector="motion", confidence=1.0, value=value)


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


def test_lost_track_still_fires_after_an_earlier_adult_sighting_has_expired():
    # Not a suppression test: by the time tick(now=31.0) runs, adult_hold_s
    # (1.0s here) has long since decayed adult_present back to False, so this
    # only shows that a stale adult sighting does not block the alert.
    # test_lost_track_bypasses_adult_suppression_but_ordinary_alerts_do_not
    # below is the one that pins the bypass guarantee itself.
    eng = engine()
    eng.handle(baby(0.0))
    eng.handle(adult(1.0))
    assert eng.adult_present
    assert [a.type for a in eng.tick(now=31.0)] == [AlertType.LOST_TRACK]


def test_lost_track_bypasses_adult_suppression_but_ordinary_alerts_do_not():
    # adult_hold_s is long enough that adult_present is still True at the
    # moment tick() runs the health checks - the earlier test above cannot
    # show this because its adult presence has already decayed by then.
    cfg = RulesConfig(
        lost_track_after_s=30.0,
        detector_silent_after_s=10.0,
        adult_min_duration_s=0.0,
        adult_hold_s=120.0,
    )
    eng = RuleEngine(config=cfg, crib_zone=CRIB)
    eng.handle(baby(0.0))
    eng.handle(adult(30.0))
    assert eng.adult_present

    alerts = eng.tick(now=31.0)
    assert eng.adult_present  # still true at the moment tick ran, not just earlier
    assert [a.type for a in alerts] == [AlertType.LOST_TRACK]

    # Contrast: under the same adult-present conditions, an ordinary
    # (non-health) alert IS suppressed - proving tick does not simply let
    # everything through.
    suppressed: list = []
    for ts, value in [(31.5, 0.3), (32.5, 0.3), (34.0, 0.3)]:
        suppressed.extend(eng.handle(motion(ts, value)))
    assert eng.adult_present
    assert suppressed == []


def test_lost_track_is_only_a_warning_while_an_adult_is_present():
    # The night-feed case: the baby is in the caregiver's arms, so no baby
    # box is detected and last_baby_seen stops advancing. The alert must
    # still fire - the monitor genuinely cannot see the baby - but at
    # WARNING, because someone is plainly there.
    cfg = RulesConfig(
        lost_track_after_s=30.0,
        detector_silent_after_s=1e9,
        adult_min_duration_s=0.0,
        adult_hold_s=120.0,
    )
    eng = RuleEngine(config=cfg, crib_zone=CRIB)
    eng.handle(baby(0.0))
    eng.handle(adult(30.0))
    assert eng.adult_present

    alerts = eng.tick(now=31.0)
    assert [a.type for a in alerts] == [AlertType.LOST_TRACK]
    assert alerts[0].severity == Severity.WARNING
    assert alerts[0].metadata["adult_present"] is True


def test_lost_track_escalates_to_critical_once_the_adult_leaves():
    cfg = RulesConfig(
        lost_track_after_s=30.0,
        detector_silent_after_s=1e9,
        adult_min_duration_s=0.0,
        adult_hold_s=5.0,
        lost_track_realert_interval_s=10.0,
    )
    eng = RuleEngine(config=cfg, crib_zone=CRIB)
    eng.handle(baby(0.0))
    eng.handle(adult(30.0))
    attended = eng.tick(now=31.0)
    assert attended[0].severity == Severity.WARNING

    # adult_hold_s expires, the baby is still not found, and the same
    # condition now reads as alone-and-unseen.
    alone = eng.tick(now=45.0)
    assert not eng.adult_present
    assert [a.type for a in alone] == [AlertType.LOST_TRACK]
    assert alone[0].severity == Severity.CRITICAL
    assert alone[0].metadata["adult_present"] is False


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


def test_a_filtered_tick_alert_is_not_latched_and_is_retried():
    """R31: latch only what actually went out.

    Today every tick alert is a health type and _allowed passes them all, so
    this is inert - which is exactly why it is worth pinning now. If a future
    filter ever drops a tick alert, latching it as "reported" would suppress
    every retry and lose the condition for good. _allowed is stubbed here to
    stand in for that filter.
    """
    eng = engine()
    eng.handle(baby(0.0))

    eng._allowed = lambda alert: False
    assert eng.tick(now=31.0) == []

    del eng._allowed  # filter lifted; the condition still holds
    alerts = eng.tick(now=32.0)
    assert [a.type for a in alerts] == [AlertType.LOST_TRACK]


def test_a_filtered_silent_detector_alert_is_not_latched_either():
    eng = engine(watched_detectors=("motion",))
    eng.tick(now=0.0)

    eng._allowed = lambda alert: False
    assert eng.tick(now=11.0) == []

    del eng._allowed
    alerts = eng.tick(now=12.0)
    assert [a.type for a in alerts] == [AlertType.DETECTOR_SILENT]


def test_lost_track_repeats_once_the_realert_interval_has_passed():
    """A blanket over the camera used to yield exactly one CRITICAL and then
    silence all night. The condition persists, so the alert must too."""
    cfg = RulesConfig(
        lost_track_after_s=30.0,
        detector_silent_after_s=1e9,
        lost_track_realert_interval_s=300.0,
    )
    eng = RuleEngine(config=cfg, crib_zone=CRIB)
    eng.handle(baby(0.0))

    first = eng.tick(now=31.0)
    assert [a.type for a in first] == [AlertType.LOST_TRACK]

    # Still lost, but well inside the re-alert interval: stay quiet.
    assert eng.tick(now=200.0) == []
    assert eng.tick(now=330.0) == []

    repeat = eng.tick(now=331.0)  # 300s after the first alert
    assert [a.type for a in repeat] == [AlertType.LOST_TRACK]
    assert repeat[0].metadata["seconds_since_last_seen"] == 331.0


def test_a_silent_detector_repeats_once_the_realert_interval_has_passed():
    cfg = RulesConfig(
        lost_track_after_s=1e9,
        detector_silent_after_s=10.0,
        detector_silent_realert_interval_s=60.0,
    )
    eng = RuleEngine(config=cfg, crib_zone=CRIB, watched_detectors=("motion",))
    eng.handle(
        MotionEnergy(ts=0.0, detector="motion", confidence=1.0, value=0.0)
    )

    assert [a.type for a in eng.tick(now=11.0)] == [AlertType.DETECTOR_SILENT]
    assert eng.tick(now=40.0) == []
    assert eng.tick(now=70.0) == []
    assert [a.type for a in eng.tick(now=71.0)] == [AlertType.DETECTOR_SILENT]


def test_a_recovered_detector_re_arms_immediately_rather_than_waiting():
    # Recovery clears the re-alert clock, so a second outage is reported at
    # detector_silent_after_s and not one re-alert interval later.
    cfg = RulesConfig(
        lost_track_after_s=1e9,
        detector_silent_after_s=10.0,
        detector_silent_realert_interval_s=1e6,
    )
    eng = RuleEngine(config=cfg, crib_zone=CRIB, watched_detectors=("motion",))
    eng.handle(
        MotionEnergy(ts=0.0, detector="motion", confidence=1.0, value=0.0)
    )
    assert len(eng.tick(now=11.0)) == 1

    eng.handle(
        MotionEnergy(ts=12.0, detector="motion", confidence=1.0, value=0.0)
    )
    assert eng.tick(now=13.0) == []
    assert [a.type for a in eng.tick(now=23.0)] == [AlertType.DETECTOR_SILENT]
