"""SQLite event store with snapshot files and a retention sweeper.

An always-on camera will fill a disk, so retention is not optional.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from babymon.events import Alert

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    type          TEXT    NOT NULL,
    severity      TEXT    NOT NULL,
    started_at    REAL    NOT NULL,
    ended_at      REAL,
    recorded_at   REAL    NOT NULL,
    confidence    REAL    NOT NULL,
    snapshot_path TEXT,
    clip_path     TEXT,
    metadata_json TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_recorded_at ON events(recorded_at);
"""


class SqliteStore:
    def __init__(
        self,
        db_path: str | Path,
        snapshot_dir: str | Path,
        retention_days: int = 7,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.db_path = Path(db_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.retention_days = retention_days
        self.now = now

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def emit(self, alert: Alert, snapshot: np.ndarray | None = None) -> int:
        recorded_at = self.now()
        snapshot_path: str | None = None
        if snapshot is not None:
            # Add random suffix to ensure uniqueness even for same type at same millisecond
            suffix = uuid.uuid4().hex[:8]
            name = f"{int(recorded_at * 1000)}_{alert.type.value}_{suffix}.jpg"
            path = self.snapshot_dir / name
            if not cv2.imwrite(str(path), snapshot):
                logger.error(f"Failed to write snapshot file: {path}")
                snapshot_path = None
            else:
                snapshot_path = str(path)

        # Safely serialize metadata, catching non-JSON-serializable values
        try:
            metadata_json = json.dumps(alert.metadata)
        except TypeError as e:
            logger.error(f"Metadata not serialisable for alert {alert.type.value}: {e}")
            metadata_json = json.dumps({
                "_error": "metadata not serialisable",
                "_repr": repr(alert.metadata)[:500]
            })

        cur = self._conn.execute(
            "INSERT INTO events (type, severity, started_at, ended_at, "
            "recorded_at, confidence, snapshot_path, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                alert.type.value,
                alert.severity.value,
                alert.started_at,
                alert.ended_at,
                recorded_at,
                alert.confidence,
                snapshot_path,
                metadata_json,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            out.append(item)
        return out

    def sweep(self) -> int:
        cutoff = self.now() - self.retention_days * 86400
        rows = self._conn.execute(
            "SELECT id, snapshot_path FROM events WHERE recorded_at < ?",
            (cutoff,),
        ).fetchall()
        for row in rows:
            if row["snapshot_path"]:
                Path(row["snapshot_path"]).unlink(missing_ok=True)
        self._conn.execute("DELETE FROM events WHERE recorded_at < ?", (cutoff,))
        self._conn.commit()
        return len(rows)

    def close(self) -> None:
        self._conn.close()
