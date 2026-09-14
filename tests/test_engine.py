from babymon.config import EventRuleConfig, RulesConfig
from babymon.events import AlertType, MotionEnergy, PersonBox
from babymon.rules.engine import RuleEngine
from babymon.rules.zones import Zone

CRIB = Zone(
    name="crib", polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
)


def rules(**overrides) -> RulesConfig:
    base = dict(
        awake=EventRuleConfig(
            enter_threshold=0.02, exit_threshold=0.005,
            min_duration_s=2.0, exit_duration_s=5.0, cooldown_s=0.0,
        ),
        zone_exit=EventRuleConfig(
            enter_threshold=0.5, exit_threshold=0.5,
            min_duration_s=2.0, exit_duration_s=3.0, cooldown_s=0.0,
        ),
        adult_min_duration_s=0.0,
        adult_hold_s=1.0,
        lost_track_after_s=1e9,
    )
    base.update(overrides)
    return RulesConfig(**base)


def engine(**overrides) -> RuleEngine:
    return RuleEngine(config=rules(**overrides), crib_zone=CRIB)


def motion(ts, value):
    return MotionEnergy(ts=ts, detector="motion", confidence=1.0, value=value)


def baby(ts, bbox=(0.45, 0.45, 0.60, 0.60)):
    return PersonBox(
        ts=ts, detector="person", confidence=0.9, bbox=bbox, label="baby"
    )


def adult(ts):
    return PersonBox(
        ts=ts, detector="person", confidence=0.9,
        bbox=(0.0, 0.0, 0.5, 0.9), label="adult",
    )


def unknown(ts, bbox=(0.45, 0.45, 0.60, 0.60)):
    return PersonBox(
        ts=ts, detector="person", confidence=0.9, bbox=bbox, label="unknown"
    )


def drain(eng, observations):
    alerts = []
    for obs in observations:
        alerts.extend(eng.handle(obs))
    return alerts


def test_sustained_motion_raises_an_awake_alert():
    eng = engine()
    alerts = drain(eng, [motion(0.0, 0.3), motion(1.0, 0.3), motion(2.5, 0.3)])
    assert [a.type for a in alerts] == [AlertType.AWAKE]
    assert alerts[0].started_at == 0.0


def test_brief_motion_does_not_alert():
    eng = engine()
    assert drain(eng, [motion(0.0, 0.3), motion(0.5, 0.0)]) == []


def test_adult_presence_suppresses_the_awake_alert():
    eng = engine()
    # A real person detector reports the adult on every frame it sees them,
    # so the stream interleaves rather than showing a single box.
    alerts = drain(
        eng,
        [
            adult(0.0), motion(0.0, 0.3),
            adult(1.0), motion(1.0, 0.3),
            adult(2.0), motion(2.5, 0.3),
        ],
    )
    assert alerts == []
    assert eng.adult_present


def test_alerts_resume_once_the_adult_leaves():
    eng = engine()
    drain(
        eng,
        [
            adult(0.0), motion(0.0, 0.3),
            adult(1.0), motion(1.0, 0.3),
            adult(2.0), motion(2.5, 0.3),
        ],
    )
    # Stillness clears the awake state, then the adult stops being reported
    # for longer than adult_hold_s and motion resumes.
    alerts = drain(
        eng,
        [
            motion(4.0, 0.0), motion(9.5, 0.0),
            baby(10.0), motion(10.0, 0.3), motion(13.0, 0.3),
        ],
    )
    assert AlertType.AWAKE in [a.type for a in alerts]
    assert not eng.adult_present


def test_baby_leaving_the_crib_zone_raises_zone_exit():
    eng = engine()
    outside = (0.85, 0.85, 0.95, 0.95)
    alerts = drain(
        eng,
        [baby(0.0), baby(1.0, outside), baby(2.0, outside), baby(3.5, outside)],
    )
    assert AlertType.ZONE_EXIT in [a.type for a in alerts]


def test_baby_inside_the_zone_does_not_raise_zone_exit():
    eng = engine()
    alerts = drain(eng, [baby(0.0), baby(2.0), baby(4.0)])
    assert alerts == []


def test_baby_observations_update_last_seen():
    eng = engine()
    eng.handle(baby(7.0))
    assert eng.last_baby_seen == 7.0


def test_unknown_labelled_boxes_do_not_suppress_or_update_last_seen():
    # Carried forward from Task 7's review: without a crib zone, PersonDetector
    # labels everyone "unknown" rather than "adult", specifically so a
    # misconfigured monitor cannot silently suppress every alert. This is the
    # end-to-end check that the fix holds at the engine boundary - UNKNOWN is
    # not SAFE, so it must not be treated as either "adult" or "baby".
    eng = engine()
    alerts = drain(
        eng,
        [unknown(0.0), unknown(1.0), unknown(2.0), unknown(3.0)],
    )
    assert alerts == []
    assert not eng.adult_present
    assert eng.last_baby_seen is None
