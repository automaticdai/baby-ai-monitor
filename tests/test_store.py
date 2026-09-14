import numpy as np
import pytest

from babymon.events import Alert, AlertType, Severity
from babymon.sinks.store import SqliteStore
from unittest.mock import patch


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


def test_imwrite_failure_logs_error_and_leaves_snapshot_path_none(store, caplog):
    """Finding 1: cv2.imwrite can fail silently; we must check its return value."""
    with patch("babymon.sinks.store.cv2.imwrite", return_value=False):
        row_id = store.emit(alert(), snapshot=image())

    # Alert row must be inserted despite snapshot write failure
    rows = store.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == row_id
    assert row["type"] == "awake"

    # snapshot_path should be None, not pointing to a nonexistent file
    assert row["snapshot_path"] is None

    # Error must be logged
    assert "Failed to write snapshot file" in caplog.text


def test_same_type_same_time_different_snapshots(store):
    """Finding 2: Multiple alerts of same type at same millisecond must have unique files."""
    # Emit two alerts of the same type at the same clock reading, both with snapshots
    row_id_1 = store.emit(alert(AlertType.AWAKE, started_at=1.0), snapshot=image())
    row_id_2 = store.emit(alert(AlertType.AWAKE, started_at=2.0), snapshot=image())

    rows = store.recent(limit=2)
    assert len(rows) == 2

    path_1 = rows[1]["snapshot_path"]  # Oldest first after DESC order
    path_2 = rows[0]["snapshot_path"]  # Newest

    # Paths must be different (random suffix ensures uniqueness)
    assert path_1 is not None
    assert path_2 is not None
    assert path_1 != path_2

    # Both files must exist
    from pathlib import Path
    assert Path(path_1).exists()
    assert Path(path_2).exists()


def test_unserialisable_metadata_logs_error_and_persists_alert(store, caplog):
    """Finding 3: Metadata might not be JSON-serializable; we must still persist the alert."""
    # Create an alert with unserialisable metadata (a set)
    bad_alert = Alert(
        type=AlertType.AWAKE,
        severity=Severity.CRITICAL,
        started_at=1.0,
        confidence=0.9,
        metadata={"unserializable": {1, 2, 3}},  # Sets are not JSON-serializable
    )

    # emit() must succeed and insert the alert row
    row_id = store.emit(bad_alert)

    # Alert row must be present
    rows = store.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == row_id
    assert row["type"] == "awake"
    assert row["severity"] == "critical"

    # Metadata should record the error, not lose the alert
    assert "_error" in row["metadata"]
    assert row["metadata"]["_error"] == "metadata not serialisable"
    assert "_repr" in row["metadata"]

    # Error must be logged
    assert "Metadata not serialisable" in caplog.text
