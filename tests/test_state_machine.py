from babymon.config import EventRuleConfig
from babymon.rules.state import EvidenceStateMachine


def machine(**overrides) -> EvidenceStateMachine:
    defaults = dict(
        enter_threshold=0.5,
        exit_threshold=0.2,
        min_duration_s=2.0,
        exit_duration_s=3.0,
        cooldown_s=0.0,
    )
    defaults.update(overrides)
    cfg = EventRuleConfig(**defaults)
    return EvidenceStateMachine(name="test", config=cfg)


def test_does_not_enter_before_the_minimum_duration():
    m = machine()
    assert m.update(0.0, 0.9) is None
    assert m.update(1.0, 0.9) is None
    assert not m.active


def test_enters_once_evidence_is_sustained():
    m = machine()
    m.update(0.0, 0.9)
    assert m.update(2.0, 0.9) == "entered"
    assert m.active
    assert m.entered_at == 0.0


def test_a_dip_below_the_enter_threshold_restarts_the_clock():
    m = machine()
    m.update(0.0, 0.9)
    m.update(1.0, 0.1)  # dip
    m.update(1.5, 0.9)  # clock restarts here
    assert m.update(3.0, 0.9) is None
    assert m.update(3.6, 0.9) == "entered"


def test_values_in_the_hysteresis_band_do_not_flap():
    m = machine()
    m.update(0.0, 0.9)
    m.update(2.0, 0.9)
    # 0.35 is below enter but above exit: state must hold.
    for ts in (3.0, 5.0, 9.0, 20.0):
        assert m.update(ts, 0.35) is None
    assert m.active


def test_exits_after_sustained_low_values():
    m = machine()
    m.update(0.0, 0.9)
    m.update(2.0, 0.9)
    m.update(3.0, 0.1)
    assert m.update(5.0, 0.1) is None  # only 2s below, needs 3s
    assert m.update(6.1, 0.1) == "exited"
    assert not m.active


def test_cooldown_suppresses_a_rapid_second_alert():
    m = machine(cooldown_s=100.0)
    m.update(0.0, 0.9)
    assert m.update(2.0, 0.9) == "entered"
    m.update(3.0, 0.1)
    m.update(6.1, 0.1)  # exits
    m.update(7.0, 0.9)
    # Re-enters as state, but within cooldown so no alert is reported.
    assert m.update(9.0, 0.9) is None
    assert m.active


def test_cooldown_expires():
    m = machine(cooldown_s=5.0)
    m.update(0.0, 0.9)
    m.update(2.0, 0.9)
    m.update(3.0, 0.1)
    m.update(6.1, 0.1)
    m.update(20.0, 0.9)
    assert m.update(22.0, 0.9) == "entered"
