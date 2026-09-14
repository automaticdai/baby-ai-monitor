"""CLI-level tests for babymon.__main__.

Kept minimal and process-free: main() is called directly with an argv list
so nothing spawns a subprocess, and every source/store that could block or
write to disk is patched to raise if it's ever constructed - so a test that
would otherwise hang (e.g. an infinite webcam retry loop) fails fast and
loudly instead.
"""

import time

import numpy as np
import pytest

from babymon import __main__ as cli
from babymon.events import Alert, AlertType, Frame, Severity


def test_replay_without_a_file_source_fails_fast(monkeypatch, capsys):
    def _must_not_construct(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"{name} must not be constructed")
        return _raise

    # No --source and no config means cfg.source.kind defaults to "webcam",
    # which is an infinite generator. If the fail-fast guard is ever removed
    # or misplaced, these patches turn the resulting hang into an immediate,
    # obvious assertion failure instead.
    monkeypatch.setattr(cli, "WebcamSource", _must_not_construct("WebcamSource"))
    monkeypatch.setattr(cli, "SqliteStore", _must_not_construct("SqliteStore"))
    monkeypatch.setattr(
        cli, "YoloPersonModel", _must_not_construct("YoloPersonModel")
    )

    rc = cli.main(["--replay"])

    assert rc != 0
    err = capsys.readouterr().err
    assert "--source" in err


class _StubStore:
    """Stands in for SqliteStore so nothing touches the disk."""

    def __init__(self, **kwargs):
        self.alerts = []

    def sweep(self):
        return 0

    def emit(self, alert, snapshot=None):
        self.alerts.append(alert)
        return len(self.alerts)

    def close(self):
        pass


def _frame(seq: int) -> Frame:
    return Frame(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        ts=time.monotonic(),
        seq=seq,
        source_id="t",
    )


class _FiniteSource:
    source_id = "t"

    def frames(self):
        for seq in range(4):
            yield _frame(seq)

    def close(self):
        pass


class _DyingSource:
    source_id = "t"

    def frames(self):
        yield _frame(0)
        raise RuntimeError("device disappeared")

    def close(self):
        pass


@pytest.fixture
def stub_store(monkeypatch):
    store = _StubStore()
    monkeypatch.setattr(cli, "SqliteStore", lambda **kwargs: store)
    monkeypatch.setattr(
        cli, "YoloPersonModel", lambda *a, **k: pytest.fail("no model here")
    )
    return store


def test_main_returns_non_zero_when_the_source_dies(monkeypatch, stub_store):
    """A monitor that went blind must not report success: a supervisor set
    to Restart=on-failure has to see a failure to act on."""
    monkeypatch.setattr(cli, "FileVideoSource", lambda *a, **k: _DyingSource())

    rc = cli.main(["--source", "clip.mp4", "--no-person"])

    assert rc != 0
    assert AlertType.SOURCE_LOST in [a.type for a in stub_store.alerts]


def test_main_returns_zero_when_a_finite_source_reaches_eof(
    monkeypatch, stub_store
):
    monkeypatch.setattr(cli, "FileVideoSource", lambda *a, **k: _FiniteSource())

    rc = cli.main(["--source", "clip.mp4", "--no-person"])

    assert rc == 0
    assert AlertType.SOURCE_LOST not in [a.type for a in stub_store.alerts]


def test_format_alert_explains_a_lost_track_rather_than_printing_a_bare_zero():
    """The first column is when the condition began, which for lost_track is
    when the baby was last seen — not when the alert fired. The detail column
    must carry the number a person actually wants."""
    alert = Alert(
        type=AlertType.LOST_TRACK,
        severity=Severity.CRITICAL,
        started_at=0.0,
        metadata={"seconds_since_last_seen": 60.0, "adult_present": False},
    )
    line = cli.format_alert(alert)
    assert "lost_track" in line
    assert "critical" in line
    assert "baby not seen for 60.0s" in line


def test_format_alert_notes_when_an_adult_is_present():
    alert = Alert(
        type=AlertType.LOST_TRACK,
        severity=Severity.WARNING,
        started_at=5.0,
        metadata={"seconds_since_last_seen": 60.0, "adult_present": True},
    )
    assert "an adult is present" in cli.format_alert(alert)


def test_format_alert_names_the_silent_detector():
    alert = Alert(
        type=AlertType.DETECTOR_SILENT,
        severity=Severity.WARNING,
        started_at=1.0,
        metadata={"detector": "motion"},
    )
    assert "'motion' stopped reporting" in cli.format_alert(alert)


def test_format_alert_reports_motion_energy_for_awake():
    alert = Alert(
        type=AlertType.AWAKE,
        severity=Severity.INFO,
        started_at=8.0,
        metadata={"motion_energy": 0.0459},
    )
    line = cli.format_alert(alert)
    assert line.startswith("    8.00s")
    assert "motion energy 0.046" in line


def test_format_alert_survives_empty_metadata():
    alert = Alert(
        type=AlertType.AWAKE, severity=Severity.INFO, started_at=2.0, metadata={}
    )
    line = cli.format_alert(alert)
    assert "awake" in line
    assert not line.endswith(" ")
