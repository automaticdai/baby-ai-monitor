import numpy as np
import pytest

from babymon.events import Alert, AlertType, Severity
from babymon.sinks.store import SqliteStore


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(
        db_path=tmp_path / "events.db",
        snapshot_dir=tmp_path / "snaps",
        retention_days=7,
        now=lambda: 1_000_000.0,
    )
    yield s
    s.close()


def alert(type_=AlertType.AWAKE, started_at=1.0) -> Alert:
    return Alert(
        type=type_,
        severity=Severity.INFO,
        started_at=started_at,
        confidence=0.8,
        metadata={"motion_energy": 0.3},
    )


def image() -> np.ndarray:
    return np.full((40, 40, 3), 128, dtype=np.uint8)


def test_emit_persists_the_alert(store):
    store.emit(alert())
    (row,) = store.recent()
    assert row["type"] == "awake"
    assert row["severity"] == "info"
    assert row["metadata"]["motion_energy"] == 0.3


def test_emit_writes_a_snapshot_file(store):
    store.emit(alert(), snapshot=image())
    (row,) = store.recent()
    assert row["snapshot_path"] is not None
    from pathlib import Path
    assert Path(row["snapshot_path"]).exists()


def test_emit_without_a_snapshot_leaves_the_path_empty(store):
    store.emit(alert())
    assert store.recent()[0]["snapshot_path"] is None


def test_recent_returns_newest_first(store):
    store.emit(alert(AlertType.AWAKE, started_at=1.0))
    store.emit(alert(AlertType.ZONE_EXIT, started_at=2.0))
    assert [r["type"] for r in store.recent()] == ["zone_exit", "awake"]


def test_recent_respects_the_limit(store):
    for i in range(5):
        store.emit(alert(started_at=float(i)))
    assert len(store.recent(limit=2)) == 2


def test_sweep_deletes_rows_older_than_retention(tmp_path):
    clock = {"t": 1_000_000.0}
    store = SqliteStore(
        db_path=tmp_path / "e.db",
        snapshot_dir=tmp_path / "s",
        retention_days=1,
        now=lambda: clock["t"],
    )
    store.emit(alert(), snapshot=image())
    old_path = store.recent()[0]["snapshot_path"]

    clock["t"] += 2 * 86400  # two days later
    assert store.sweep() == 1
    assert store.recent() == []

    from pathlib import Path
    assert not Path(old_path).exists()
    store.close()


def test_sweep_keeps_rows_inside_retention(store):
    store.emit(alert())
    assert store.sweep() == 0
    assert len(store.recent()) == 1
