# Core Pipeline & Rule Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local video pipeline, motion and person detectors, and the rule engine that turns raw observations into debounced safety alerts stored in SQLite.

**Architecture:** One process, staged pipeline. A capture thread writes frames into a latest-wins buffer; detector worker threads read the newest frame at their own stride and publish `Observation`s to an in-process event bus; a single-threaded rule engine consumes them in timestamp order and emits `Alert`s to sinks. Observations are stateless facts; only the rule engine creates alerts.

**Tech Stack:** Python 3.11+, numpy, OpenCV (`opencv-python`), pydantic v2, PyYAML, ultralytics (YOLO11n), pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-baby-ai-monitor-design.md`

## Global Constraints

- Python `>=3.11`. Package is a flat `babymon/` package at the repo root; tests in `tests/`.
- **All inference is local.** No cloud APIs, no network egress anywhere in this plan.
- **Only `Alert` objects reach a user.** No detector notifies anyone directly; detectors return `Observation`s and nothing else.
- **All spatial coordinates are normalised to 0..1** (both bounding boxes and zone polygons), so configuration is independent of camera resolution.
- **`UNKNOWN` is never `SAFE`.** Absence of detections must produce an alert, never silence.
- **System-health alerts are never suppressed** by `adult_present` or by any other suppression rule.
- Timestamps are floats in seconds from a monotonic clock. Every component that needs the current time takes a `now: Callable[[], float]` parameter defaulting to `time.monotonic`, so tests inject a fake clock.
- This is an assistive monitor, **not a medical device**. Never write copy claiming it detects or prevents SIDS.

---

## File Structure

| File | Responsibility |
|---|---|
| `babymon/events.py` | `Frame`, `Observation` subclasses, `Alert`, enums. No logic beyond derived geometry. |
| `babymon/bus.py` | `EventBus` pub/sub + `ReorderBuffer` for timestamp ordering. |
| `babymon/config.py` | Pydantic config models, YAML loading, validation. |
| `babymon/sources/base.py` | `VideoSource` protocol, `FrameBuffer` (latest-wins). |
| `babymon/sources/file.py` | Deterministic video-file replay. |
| `babymon/sources/webcam.py` | Live webcam capture. |
| `babymon/rules/zones.py` | `Zone` polygon geometry. |
| `babymon/detectors/base.py` | `Detector` protocol. |
| `babymon/detectors/motion.py` | Frame-differencing motion energy. |
| `babymon/detectors/person.py` | YOLO person detection + baby/adult attribution. |
| `babymon/rules/state.py` | `EvidenceStateMachine`: hysteresis, min duration, cooldown. |
| `babymon/rules/engine.py` | Routes observations to machines, applies suppression, lost-track, watchdog. |
| `babymon/sinks/base.py` | `Sink` protocol. |
| `babymon/sinks/store.py` | SQLite + snapshot persistence + retention sweep. |
| `babymon/runner.py` | Thread wiring, detector isolation, lifecycle. |
| `babymon/__main__.py` | CLI entry point. |

---

### Task 1: Project scaffolding and core types

**Files:**
- Create: `pyproject.toml`, `babymon/__init__.py`, `babymon/events.py`, `.gitignore`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Frame(image, ts, seq, source_id)`; `Observation(ts, detector, confidence)` base; `PersonBox(..., bbox, label)` with `.center` and `.area` properties; `MotionEnergy(..., value)`; `Pose(..., keypoints)`; `CryProbability(..., value)`; `AlertType`, `Severity`, `Alert(type, severity, started_at, ended_at=None, confidence=1.0, metadata={})`.

- [ ] **Step 1: Create the project scaffolding**

`pyproject.toml`:

```toml
[project]
name = "babymon"
version = "0.1.0"
description = "Local baby monitor with real-time safety alerting"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "opencv-python>=4.9",
    "pydantic>=2.6",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
detect = ["ultralytics>=8.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["babymon"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:

```
__pycache__/
*.py[cod]
.venv/
data/
*.egg-info/
.pytest_cache/
```

Then create the empty package and install:

```bash
mkdir -p babymon tests
touch babymon/__init__.py tests/__init__.py
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 2: Write the failing test**

`tests/test_events.py`:

```python
import numpy as np
import pytest

from babymon.events import (
    Alert,
    AlertType,
    Frame,
    MotionEnergy,
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.events'`

- [ ] **Step 4: Write minimal implementation**

`babymon/events.py`:

```python
"""Core data types.

Three kinds of thing, deliberately kept distinct:

* ``Frame``       - an image at an instant.
* ``Observation`` - what one detector saw at an instant. Stateless, no
                    interpretation, always carries a confidence.
* ``Alert``       - a decision the rule engine reached over time. The only
                    type that ever reaches a user.

All spatial coordinates are normalised to 0..1 so that configuration does not
depend on camera resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2, normalised


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    ts: float
    seq: int
    source_id: str


@dataclass(frozen=True)
class Observation:
    ts: float
    detector: str
    confidence: float


@dataclass(frozen=True)
class PersonBox(Observation):
    bbox: BBox
    label: str = "unknown"  # "baby" | "adult" | "unknown"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class MotionEnergy(Observation):
    value: float  # 0..1, fraction of zone pixels that changed


@dataclass(frozen=True)
class Pose(Observation):
    # keypoint name -> (x, y, visibility)
    keypoints: dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class CryProbability(Observation):
    value: float  # 0..1


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    AWAKE = "awake"
    CRYING = "crying"
    PRONE = "prone"
    FACE_COVERED = "face_covered"
    ZONE_EXIT = "zone_exit"
    LOST_TRACK = "lost_track"
    DETECTOR_SILENT = "detector_silent"
    SOURCE_LOST = "source_lost"
    CAMERA_MOVED = "camera_moved"


#: Alerts about the monitor itself. Never suppressed - a monitor that has
#: failed must be noisy about it.
HEALTH_ALERTS = frozenset(
    {
        AlertType.LOST_TRACK,
        AlertType.DETECTOR_SILENT,
        AlertType.SOURCE_LOST,
        AlertType.CAMERA_MOVED,
    }
)


@dataclass(frozen=True)
class Alert:
    type: AlertType
    severity: Severity
    started_at: float
    ended_at: float | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_events.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore babymon/__init__.py babymon/events.py tests/__init__.py tests/test_events.py
git commit -m "feat: add project scaffolding and core event types"
```

---

### Task 2: Event bus and reorder buffer

**Files:**
- Create: `babymon/bus.py`
- Test: `tests/test_bus.py`

**Interfaces:**
- Consumes: `Observation` from Task 1.
- Produces: `EventBus.publish(obs)`, `EventBus.subscribe() -> Subscription`, `Subscription.get(timeout) -> Observation | None`; `ReorderBuffer(window_s).add(obs)` and `.drain(now) -> list[Observation]` returning timestamp-sorted observations whose `ts + window_s <= now`.

- [ ] **Step 1: Write the failing test**

`tests/test_bus.py`:

```python
from babymon.bus import EventBus, ReorderBuffer
from babymon.events import MotionEnergy


def motion(ts: float, value: float = 0.5) -> MotionEnergy:
    return MotionEnergy(ts=ts, detector="motion", confidence=1.0, value=value)


def test_subscriber_receives_published_observations():
    bus = EventBus()
    sub = bus.subscribe()
    obs = motion(1.0)
    bus.publish(obs)
    assert sub.get(timeout=0.1) is obs


def test_every_subscriber_gets_its_own_copy():
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    obs = motion(1.0)
    bus.publish(obs)
    assert a.get(timeout=0.1) is obs
    assert b.get(timeout=0.1) is obs


def test_get_returns_none_when_nothing_published():
    assert EventBus().subscribe().get(timeout=0.01) is None


def test_reorder_buffer_holds_observations_inside_the_window():
    buf = ReorderBuffer(window_s=0.5)
    buf.add(motion(2.0))
    assert buf.drain(now=2.1) == []


def test_reorder_buffer_releases_in_timestamp_order():
    buf = ReorderBuffer(window_s=0.5)
    buf.add(motion(2.0))
    buf.add(motion(1.0))
    released = buf.drain(now=2.6)
    assert [o.ts for o in released] == [1.0, 2.0]


def test_reorder_buffer_does_not_re_release():
    buf = ReorderBuffer(window_s=0.5)
    buf.add(motion(1.0))
    assert len(buf.drain(now=2.0)) == 1
    assert buf.drain(now=3.0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.bus'`

- [ ] **Step 3: Write minimal implementation**

`babymon/bus.py`:

```python
"""In-process pub/sub.

This module is the designated seam: swapping ``EventBus`` for a ZeroMQ or
Redis implementation splits the pipeline across machines without touching
detectors or rules.
"""

from __future__ import annotations

import queue
import threading

from babymon.events import Observation


class Subscription:
    def __init__(self, maxsize: int = 1000) -> None:
        self._q: queue.Queue[Observation] = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def put(self, obs: Observation) -> None:
        try:
            self._q.put_nowait(obs)
        except queue.Full:
            # A subscriber that cannot keep up loses observations rather than
            # stalling every detector behind it.
            self.dropped += 1

    def get(self, timeout: float = 1.0) -> Observation | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()

    def subscribe(self, maxsize: int = 1000) -> Subscription:
        sub = Subscription(maxsize=maxsize)
        with self._lock:
            self._subs.append(sub)
        return sub

    def publish(self, obs: Observation) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub.put(obs)


class ReorderBuffer:
    """Holds observations briefly so they can be consumed in timestamp order.

    Detector threads finish out of order, but the rule engine's determinism
    depends on seeing observations sorted by ``ts``. Holding each observation
    for ``window_s`` before release buys that ordering for a bounded latency
    cost.
    """

    def __init__(self, window_s: float = 0.25) -> None:
        self.window_s = window_s
        self._pending: list[Observation] = []

    def add(self, obs: Observation) -> None:
        self._pending.append(obs)

    def drain(self, now: float) -> list[Observation]:
        ready = [o for o in self._pending if o.ts + self.window_s <= now]
        self._pending = [o for o in self._pending if o.ts + self.window_s > now]
        return sorted(ready, key=lambda o: o.ts)

    def flush(self) -> list[Observation]:
        """Release everything, for end-of-stream in replay."""
        ready, self._pending = sorted(self._pending, key=lambda o: o.ts), []
        return ready
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bus.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add babymon/bus.py tests/test_bus.py
git commit -m "feat: add event bus and timestamp reorder buffer"
```

---

### Task 3: Configuration

**Files:**
- Create: `babymon/config.py`, `config.example.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Config.load(path) -> Config` with attributes `source`, `zones`, `rules`, `detectors`, `storage`; `EventRuleConfig(enter_threshold, exit_threshold, min_duration_s, exit_duration_s, cooldown_s)`; `RulesConfig(awake, zone_exit, adult_min_duration_s, adult_hold_s, lost_track_after_s, detector_silent_after_s, reorder_window_s)`; `ZoneConfig(name, polygon)`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError

from babymon.config import Config, ZoneConfig


def test_defaults_load_without_a_file():
    cfg = Config()
    assert cfg.source.kind == "webcam"
    assert cfg.rules.lost_track_after_s == 60.0
    assert cfg.storage.retention_days == 7


def test_load_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "source:\n"
        "  kind: file\n"
        "  path: clip.mp4\n"
        "zones:\n"
        "  - name: crib\n"
        "    polygon: [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]\n"
        "rules:\n"
        "  lost_track_after_s: 30\n"
    )
    cfg = Config.load(path)
    assert cfg.source.kind == "file"
    assert cfg.source.path == "clip.mp4"
    assert cfg.zones[0].name == "crib"
    assert cfg.rules.lost_track_after_s == 30.0


def test_polygon_needs_at_least_three_points():
    with pytest.raises(ValidationError):
        ZoneConfig(name="crib", polygon=[(0.1, 0.1), (0.9, 0.9)])


def test_polygon_coordinates_must_be_normalised():
    with pytest.raises(ValidationError):
        ZoneConfig(name="crib", polygon=[(0, 0), (2.0, 0), (1, 1)])


def test_file_source_requires_a_path():
    with pytest.raises(ValidationError):
        Config(source={"kind": "file"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.config'`

- [ ] **Step 3: Write minimal implementation**

`babymon/config.py`:

```python
"""Configuration. Invalid config fails loudly at startup, not at 3am."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ZoneConfig(BaseModel):
    name: str
    polygon: list[tuple[float, float]] = Field(min_length=3)

    @field_validator("polygon")
    @classmethod
    def _normalised(
        cls, v: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"polygon point ({x}, {y}) is outside 0..1; "
                    "zone coordinates are normalised"
                )
        return v


class SourceConfig(BaseModel):
    kind: Literal["webcam", "file"] = "webcam"
    device: int = 0
    path: str | None = None
    realtime: bool = True
    source_id: str = "cam0"

    @model_validator(mode="after")
    def _path_required_for_file(self) -> "SourceConfig":
        if self.kind == "file" and not self.path:
            raise ValueError("source.path is required when source.kind is 'file'")
        return self


class EventRuleConfig(BaseModel):
    enter_threshold: float = 0.5
    exit_threshold: float = 0.3
    min_duration_s: float = 2.0
    exit_duration_s: float = 3.0
    cooldown_s: float = 60.0

    @model_validator(mode="after")
    def _hysteresis_ordered(self) -> "EventRuleConfig":
        if self.exit_threshold > self.enter_threshold:
            raise ValueError(
                "exit_threshold must not exceed enter_threshold; "
                "hysteresis requires enter >= exit"
            )
        return self


class RulesConfig(BaseModel):
    awake: EventRuleConfig = Field(
        default_factory=lambda: EventRuleConfig(
            enter_threshold=0.02, exit_threshold=0.005,
            min_duration_s=2.0, exit_duration_s=10.0, cooldown_s=120.0,
        )
    )
    zone_exit: EventRuleConfig = Field(
        default_factory=lambda: EventRuleConfig(
            enter_threshold=0.5, exit_threshold=0.5,
            min_duration_s=2.0, exit_duration_s=3.0, cooldown_s=60.0,
        )
    )
    # Adult presence is not a thresholded signal, it is a recency question:
    # how long since we last saw an adult, and did we see one long enough to
    # believe it. Hence plain durations rather than an EventRuleConfig.
    adult_min_duration_s: float = 1.0
    adult_hold_s: float = 5.0
    lost_track_after_s: float = 60.0
    detector_silent_after_s: float = 15.0
    reorder_window_s: float = 0.25


class DetectorsConfig(BaseModel):
    person_conf_threshold: float = 0.4
    person_stride: int = 1
    baby_max_area: float = 0.25
    motion_stride: int = 1
    motion_pixel_threshold: int = 25
    model_path: str = "yolo11n.pt"


class StorageConfig(BaseModel):
    db_path: str = "data/events.db"
    snapshot_dir: str = "data/snapshots"
    retention_days: int = 7


class Config(BaseModel):
    source: SourceConfig = Field(default_factory=SourceConfig)
    zones: list[ZoneConfig] = Field(default_factory=list)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    detectors: DetectorsConfig = Field(default_factory=DetectorsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def zone(self, name: str) -> ZoneConfig | None:
        return next((z for z in self.zones if z.name == name), None)
```

`config.example.yaml`:

```yaml
# Baby monitor configuration. All coordinates are normalised to 0..1.
source:
  kind: webcam        # webcam | file
  device: 0
  realtime: true

zones:
  - name: crib
    polygon: [[0.15, 0.20], [0.85, 0.20], [0.85, 0.90], [0.15, 0.90]]

detectors:
  person_conf_threshold: 0.4
  baby_max_area: 0.25   # person boxes larger than this are treated as adults
  motion_pixel_threshold: 25

rules:
  # Thresholds below are starting points only. They MUST be tuned against
  # recorded footage before being trusted.
  awake:
    enter_threshold: 0.02
    exit_threshold: 0.005
    min_duration_s: 2.0
    exit_duration_s: 10.0
    cooldown_s: 120.0
  lost_track_after_s: 60.0
  detector_silent_after_s: 15.0

storage:
  db_path: data/events.db
  snapshot_dir: data/snapshots
  retention_days: 7
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add babymon/config.py config.example.yaml tests/test_config.py
git commit -m "feat: add validated YAML configuration"
```

---

### Task 4: Video sources and the latest-wins frame buffer

**Files:**
- Create: `babymon/sources/__init__.py`, `babymon/sources/base.py`, `babymon/sources/file.py`, `babymon/sources/webcam.py`
- Test: `tests/test_sources.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `Frame` from Task 1.
- Produces: `FrameBuffer()` with `.put(frame)`, `.get_latest(last_seq=None, timeout=1.0) -> Frame | None`, `.drops` (int); `VideoSource` protocol with `source_id: str`, `frames() -> Iterator[Frame]`, `close()`; `FileVideoSource(path, source_id="file", realtime=True, start_ts=0.0)`; `WebcamSource(device=0, source_id="cam0", now=time.monotonic)`.

- [ ] **Step 1: Write the failing test**

`tests/conftest.py`:

```python
import cv2
import numpy as np
import pytest


def write_video(path, frames, fps=10):
    """Write a list of BGR arrays to an AVI file (MJPG is the most portable
    encoder available with opencv-python wheels)."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
    )
    assert writer.isOpened(), "could not open VideoWriter"
    for f in frames:
        writer.write(f)
    writer.release()
    return path


@pytest.fixture
def still_video(tmp_path):
    """Five identical dark frames."""
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
    return write_video(tmp_path / "still.avi", frames)
```

`tests/test_sources.py`:

```python
import numpy as np
import pytest

from babymon.events import Frame
from babymon.sources.base import FrameBuffer
from babymon.sources.file import FileVideoSource


def make_frame(seq: int) -> Frame:
    return Frame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        ts=float(seq),
        seq=seq,
        source_id="t",
    )


def test_buffer_returns_the_latest_frame():
    buf = FrameBuffer()
    buf.put(make_frame(1))
    assert buf.get_latest(timeout=0.1).seq == 1


def test_buffer_overwrites_unconsumed_frames_and_counts_drops():
    buf = FrameBuffer()
    buf.put(make_frame(1))
    buf.put(make_frame(2))
    assert buf.drops == 1
    assert buf.get_latest(timeout=0.1).seq == 2


def test_buffer_returns_none_when_no_newer_frame_arrives():
    buf = FrameBuffer()
    buf.put(make_frame(1))
    assert buf.get_latest(timeout=0.1).seq == 1
    assert buf.get_latest(last_seq=1, timeout=0.05) is None


def test_file_source_yields_sequential_frames(still_video):
    src = FileVideoSource(still_video, realtime=False)
    frames = list(src.frames())
    assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
    assert frames[0].image.shape == (240, 320, 3)


def test_file_source_timestamps_are_deterministic(still_video):
    src = FileVideoSource(still_video, realtime=False, start_ts=100.0)
    frames = list(src.frames())
    # 10 fps fixture -> 0.1s apart, derived from frame index not wall clock.
    assert frames[0].ts == pytest.approx(100.0)
    assert frames[2].ts == pytest.approx(100.2)


def test_replaying_the_same_file_gives_identical_timestamps(still_video):
    a = [f.ts for f in FileVideoSource(still_video, realtime=False).frames()]
    b = [f.ts for f in FileVideoSource(still_video, realtime=False).frames()]
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.sources'`

- [ ] **Step 3: Write the frame buffer and source protocol**

`babymon/sources/__init__.py` — empty file.

`babymon/sources/base.py`:

```python
"""Video source protocol and the latest-wins frame buffer."""

from __future__ import annotations

import threading
from typing import Iterator, Protocol

from babymon.events import Frame


class VideoSource(Protocol):
    source_id: str

    def frames(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class FrameBuffer:
    """Holds only the newest frame.

    Deliberately not a queue: for real-time alerting a stale frame has no
    value, so under load we drop old frames rather than accumulate a backlog
    that pushes every alert further behind live. ``drops`` is a health metric.
    """

    def __init__(self) -> None:
        self._frame: Frame | None = None
        self._consumed = True
        self._drops = 0
        self._cond = threading.Condition()

    @property
    def drops(self) -> int:
        with self._cond:
            return self._drops

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None and not self._consumed:
                self._drops += 1
            self._frame = frame
            self._consumed = False
            self._cond.notify_all()

    def get_latest(
        self, last_seq: int | None = None, timeout: float = 1.0
    ) -> Frame | None:
        with self._cond:
            def ready() -> bool:
                return self._frame is not None and (
                    last_seq is None or self._frame.seq != last_seq
                )

            if not ready():
                self._cond.wait_for(ready, timeout=timeout)
            if not ready():
                return None
            self._consumed = True
            return self._frame
```

- [ ] **Step 4: Write the file and webcam sources**

`babymon/sources/file.py`:

```python
"""Deterministic replay of a recorded video file.

Timestamps are derived from the frame index and the file's FPS, never from the
wall clock, so replaying the same file always produces the same timeline. That
property is what makes golden replay tests and threshold tuning possible.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2

from babymon.events import Frame


class FileVideoSource:
    def __init__(
        self,
        path: str | Path,
        source_id: str = "file",
        realtime: bool = True,
        start_ts: float = 0.0,
    ) -> None:
        self.path = str(path)
        self.source_id = source_id
        self.realtime = realtime
        self.start_ts = start_ts
        self._cap: cv2.VideoCapture | None = None

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video file: {self.path}")
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = 1.0 / fps
        seq = 0
        wall_start = time.monotonic()
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    return
                ts = self.start_ts + seq * interval
                if self.realtime:
                    target = wall_start + seq * interval
                    delay = target - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                yield Frame(image=image, ts=ts, seq=seq, source_id=self.source_id)
                seq += 1
        finally:
            cap.release()
            self._cap = None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
```

`babymon/sources/webcam.py`:

```python
"""Live webcam capture with reconnect-on-failure."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterator

import cv2

from babymon.events import Frame

log = logging.getLogger(__name__)


class WebcamSource:
    def __init__(
        self,
        device: int = 0,
        source_id: str = "cam0",
        now: Callable[[], float] = time.monotonic,
        max_backoff_s: float = 30.0,
    ) -> None:
        self.device = device
        self.source_id = source_id
        self.now = now
        self.max_backoff_s = max_backoff_s
        self._cap: cv2.VideoCapture | None = None
        self._stopped = False
        self.consecutive_failures = 0

    def frames(self) -> Iterator[Frame]:
        seq = 0
        backoff = 1.0
        while not self._stopped:
            cap = cv2.VideoCapture(self.device)
            if not cap.isOpened():
                self.consecutive_failures += 1
                log.warning(
                    "camera %s unavailable, retrying in %.1fs",
                    self.device, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                continue
            self._cap = cap
            backoff = 1.0
            self.consecutive_failures = 0
            while not self._stopped:
                ok, image = cap.read()
                if not ok:
                    log.warning("camera %s read failed, reconnecting", self.device)
                    self.consecutive_failures += 1
                    break
                yield Frame(
                    image=image, ts=self.now(), seq=seq, source_id=self.source_id
                )
                seq += 1
            cap.release()
            self._cap = None

    def close(self) -> None:
        self._stopped = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_sources.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 6: Commit**

```bash
git add babymon/sources tests/test_sources.py tests/conftest.py
git commit -m "feat: add video sources and latest-wins frame buffer"
```

---

### Task 5: Zone geometry

**Files:**
- Create: `babymon/rules/__init__.py`, `babymon/rules/zones.py`
- Test: `tests/test_zones.py`

**Interfaces:**
- Consumes: `ZoneConfig` from Task 3.
- Produces: `Zone(name, polygon)` with `.contains((x, y)) -> bool`, `.mask(height, width) -> np.ndarray` (uint8, 255 inside), `Zone.from_config(zone_config) -> Zone`.

- [ ] **Step 1: Write the failing test**

`tests/test_zones.py`:

```python
import numpy as np

from babymon.config import ZoneConfig
from babymon.rules.zones import Zone

SQUARE = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]


def test_point_inside_polygon():
    assert Zone(name="crib", polygon=SQUARE).contains((0.5, 0.5))


def test_point_outside_polygon():
    assert not Zone(name="crib", polygon=SQUARE).contains((0.05, 0.5))


def test_point_outside_on_the_other_axis():
    assert not Zone(name="crib", polygon=SQUARE).contains((0.5, 0.95))


def test_concave_polygon_excludes_the_notch():
    # An L-shape occupying the left column and bottom row.
    l_shape = [(0.0, 0.0), (0.4, 0.0), (0.4, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)]
    zone = Zone(name="l", polygon=l_shape)
    assert zone.contains((0.2, 0.2))
    assert not zone.contains((0.8, 0.2))


def test_mask_marks_interior_pixels():
    mask = Zone(name="crib", polygon=SQUARE).mask(100, 100)
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 255
    assert mask[1, 1] == 0
    assert mask.dtype == np.uint8


def test_from_config():
    zone = Zone.from_config(ZoneConfig(name="crib", polygon=SQUARE))
    assert zone.name == "crib"
    assert zone.contains((0.5, 0.5))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_zones.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.rules'`

- [ ] **Step 3: Write minimal implementation**

`babymon/rules/__init__.py` — empty file.

`babymon/rules/zones.py`:

```python
"""Region-of-interest polygons in normalised coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from babymon.config import ZoneConfig


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: list[tuple[float, float]]

    @classmethod
    def from_config(cls, cfg: ZoneConfig) -> "Zone":
        return cls(name=cfg.name, polygon=[tuple(p) for p in cfg.polygon])

    def contains(self, point: tuple[float, float]) -> bool:
        """Ray-casting point-in-polygon.

        Points exactly on an edge are not guaranteed to be inside; callers
        should not depend on boundary behaviour, and the rule layer's
        hysteresis makes single-pixel boundary cases irrelevant anyway.
        """
        x, y = point
        inside = False
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_at_y:
                    inside = not inside
        return inside

    def mask(self, height: int, width: int) -> np.ndarray:
        pts = np.array(
            [[int(x * width), int(y * height)] for x, y in self.polygon],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_zones.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add babymon/rules tests/test_zones.py
git commit -m "feat: add zone polygon geometry"
```

---

### Task 6: Motion detector

**Files:**
- Create: `babymon/detectors/__init__.py`, `babymon/detectors/base.py`, `babymon/detectors/motion.py`
- Test: `tests/test_motion_detector.py`

**Interfaces:**
- Consumes: `Frame`, `MotionEnergy` (Task 1), `Zone` (Task 5).
- Produces: `Detector` protocol with `name: str` and `process(frame: Frame) -> list[Observation]`; `MotionDetector(zone=None, pixel_threshold=25, blur_ksize=5)` with `name = "motion"`.

- [ ] **Step 1: Write the failing test**

`tests/test_motion_detector.py`:

```python
import numpy as np

from babymon.detectors.motion import MotionDetector
from babymon.events import Frame
from babymon.rules.zones import Zone

CRIB = Zone(name="crib", polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])


def frame(seq: int, image: np.ndarray) -> Frame:
    return Frame(image=image, ts=float(seq), seq=seq, source_id="t")


def blank() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_first_frame_produces_no_observation():
    assert MotionDetector().process(frame(0, blank())) == []


def test_identical_frames_produce_near_zero_energy():
    det = MotionDetector()
    det.process(frame(0, blank()))
    (obs,) = det.process(frame(1, blank()))
    assert obs.value == 0.0
    assert obs.detector == "motion"
    assert obs.ts == 1.0


def test_changed_pixels_raise_motion_energy():
    det = MotionDetector()
    det.process(frame(0, blank()))
    moved = blank()
    moved[10:60, 10:60] = 255  # a quarter of the image
    (obs,) = det.process(frame(1, moved))
    assert obs.value > 0.2


def test_motion_outside_the_zone_is_ignored():
    det = MotionDetector(zone=CRIB)  # zone is the left half
    det.process(frame(0, blank()))
    moved = blank()
    moved[10:90, 60:95] = 255  # entirely in the right half
    (obs,) = det.process(frame(1, moved))
    assert obs.value == 0.0


def test_motion_inside_the_zone_is_counted():
    det = MotionDetector(zone=CRIB)
    det.process(frame(0, blank()))
    moved = blank()
    moved[10:90, 5:45] = 255  # left half, inside the zone
    (obs,) = det.process(frame(1, moved))
    assert obs.value > 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_motion_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.detectors'`

- [ ] **Step 3: Write minimal implementation**

`babymon/detectors/__init__.py` — empty file.

`babymon/detectors/base.py`:

```python
"""Detector protocol.

A detector reports what it saw and nothing more. It never decides whether
something is worth waking a parent for - that is the rule engine's job.
"""

from __future__ import annotations

from typing import Protocol

from babymon.events import Frame, Observation


class Detector(Protocol):
    name: str

    def process(self, frame: Frame) -> list[Observation]: ...
```

`babymon/detectors/motion.py`:

```python
"""Frame-differencing motion energy, optionally restricted to a zone."""

from __future__ import annotations

import cv2
import numpy as np

from babymon.events import Frame, MotionEnergy, Observation
from babymon.rules.zones import Zone


class MotionDetector:
    name = "motion"

    def __init__(
        self,
        zone: Zone | None = None,
        pixel_threshold: int = 25,
        blur_ksize: int = 5,
    ) -> None:
        self.zone = zone
        self.pixel_threshold = pixel_threshold
        self.blur_ksize = blur_ksize
        self._prev: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._mask_shape: tuple[int, int] | None = None

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

    def _zone_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        if self.zone is None:
            return None
        if self._mask is None or self._mask_shape != shape:
            self._mask = self.zone.mask(shape[0], shape[1])
            self._mask_shape = shape
        return self._mask

    def process(self, frame: Frame) -> list[Observation]:
        current = self._prepare(frame.image)
        prev, self._prev = self._prev, current
        if prev is None or prev.shape != current.shape:
            return []

        diff = cv2.absdiff(prev, current)
        _, changed = cv2.threshold(
            diff, self.pixel_threshold, 255, cv2.THRESH_BINARY
        )

        mask = self._zone_mask(current.shape[:2])
        if mask is not None:
            changed = cv2.bitwise_and(changed, changed, mask=mask)
            total = int(np.count_nonzero(mask))
        else:
            total = changed.size

        value = 0.0 if total == 0 else float(np.count_nonzero(changed)) / total
        return [
            MotionEnergy(
                ts=frame.ts, detector=self.name, confidence=1.0, value=value
            )
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_motion_detector.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add babymon/detectors tests/test_motion_detector.py
git commit -m "feat: add zone-aware motion detector"
```

---

### Task 7: Person detector and baby/adult attribution

**Files:**
- Create: `babymon/detectors/person.py`
- Test: `tests/test_person_detector.py`

**Interfaces:**
- Consumes: `Frame`, `PersonBox` (Task 1), `Zone` (Task 5).
- Produces: `PersonModel` protocol with `detect_persons(image) -> list[tuple[BBox, float]]` (normalised bbox, confidence); `YoloPersonModel(model_path="yolo11n.pt")`; `PersonDetector(model, crib_zone=None, conf_threshold=0.4, baby_max_area=0.25)` with `name = "person"`, emitting `PersonBox` with `label` in `{"baby", "adult"}`.

- [ ] **Step 1: Write the failing test**

`tests/test_person_detector.py`:

```python
import numpy as np

from babymon.detectors.person import PersonDetector
from babymon.events import Frame
from babymon.rules.zones import Zone

CRIB = Zone(
    name="crib", polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
)


class FakeModel:
    """Stands in for YOLO so detector logic is tested without a model file."""

    def __init__(self, detections):
        self.detections = detections

    def detect_persons(self, image):
        return self.detections


def frame() -> Frame:
    return Frame(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        ts=1.0,
        seq=0,
        source_id="t",
    )


def detector(detections, **kwargs) -> PersonDetector:
    return PersonDetector(model=FakeModel(detections), crib_zone=CRIB, **kwargs)


def test_small_person_inside_the_crib_is_labelled_baby():
    det = detector([((0.45, 0.45, 0.60, 0.60), 0.9)])
    (obs,) = det.process(frame())
    assert obs.label == "baby"
    assert obs.detector == "person"


def test_large_person_inside_the_crib_is_labelled_adult():
    # Centre is inside the crib polygon but the box is far too big for a baby.
    det = detector([((0.05, 0.05, 0.95, 0.95), 0.9)])
    (obs,) = det.process(frame())
    assert obs.label == "adult"


def test_person_outside_the_crib_zone_is_labelled_adult():
    det = detector([((0.02, 0.02, 0.12, 0.12), 0.9)])
    (obs,) = det.process(frame())
    assert obs.label == "adult"


def test_low_confidence_detections_are_dropped():
    det = detector([((0.45, 0.45, 0.60, 0.60), 0.2)], conf_threshold=0.4)
    assert det.process(frame()) == []


def test_without_a_crib_zone_everything_is_unknown_sized():
    det = PersonDetector(
        model=FakeModel([((0.45, 0.45, 0.60, 0.60), 0.9)]), crib_zone=None
    )
    (obs,) = det.process(frame())
    assert obs.label == "adult"


def test_multiple_detections_are_all_reported():
    det = detector(
        [((0.45, 0.45, 0.60, 0.60), 0.9), ((0.01, 0.01, 0.9, 0.99), 0.8)]
    )
    labels = sorted(o.label for o in det.process(frame()))
    assert labels == ["adult", "baby"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_person_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.detectors.person'`

- [ ] **Step 3: Write minimal implementation**

`babymon/detectors/person.py`:

```python
"""Person detection plus baby/adult attribution.

Attribution is a geometric heuristic, not a model: nothing off the shelf
classifies "infant". A person box whose centre lies inside the crib zone and
whose area is below a threshold is the baby; anything else is an adult.

This holds for a fixed camera over a crib and breaks if the camera moves,
which is why the rule engine watches for large shifts in scene geometry and
warns rather than silently mis-attributing every later detection.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from babymon.events import BBox, Frame, Observation, PersonBox
from babymon.rules.zones import Zone


class PersonModel(Protocol):
    def detect_persons(self, image: np.ndarray) -> list[tuple[BBox, float]]: ...


class YoloPersonModel:
    """Ultralytics YOLO adapter. Returns normalised boxes for class 0 only."""

    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self._model = YOLO(model_path)

    def detect_persons(self, image: np.ndarray) -> list[tuple[BBox, float]]:
        height, width = image.shape[:2]
        results = self._model.predict(
            image, classes=[0], verbose=False
        )  # class 0 == person
        out: list[tuple[BBox, float]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                out.append(
                    (
                        (x1 / width, y1 / height, x2 / width, y2 / height),
                        float(box.conf[0]),
                    )
                )
        return out


class PersonDetector:
    name = "person"

    def __init__(
        self,
        model: PersonModel,
        crib_zone: Zone | None = None,
        conf_threshold: float = 0.4,
        baby_max_area: float = 0.25,
    ) -> None:
        self.model = model
        self.crib_zone = crib_zone
        self.conf_threshold = conf_threshold
        self.baby_max_area = baby_max_area

    def _label(self, box: PersonBox) -> str:
        if self.crib_zone is None:
            return "adult"
        in_crib = self.crib_zone.contains(box.center)
        return "baby" if in_crib and box.area <= self.baby_max_area else "adult"

    def process(self, frame: Frame) -> list[Observation]:
        out: list[Observation] = []
        for bbox, conf in self.model.detect_persons(frame.image):
            if conf < self.conf_threshold:
                continue
            box = PersonBox(
                ts=frame.ts, detector=self.name, confidence=conf, bbox=bbox
            )
            out.append(
                PersonBox(
                    ts=box.ts,
                    detector=box.detector,
                    confidence=box.confidence,
                    bbox=box.bbox,
                    label=self._label(box),
                )
            )
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_person_detector.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add babymon/detectors/person.py tests/test_person_detector.py
git commit -m "feat: add person detector with baby/adult attribution heuristic"
```

---

### Task 8: Evidence state machine

**Files:**
- Create: `babymon/rules/state.py`
- Test: `tests/test_state_machine.py`

**Interfaces:**
- Consumes: `EventRuleConfig` from Task 3.
- Produces: `EvidenceStateMachine(name, config)` with `.update(ts, value) -> "entered" | "exited" | None`, `.active` (bool), `.entered_at` (float | None). A transition suppressed by cooldown still changes `.active` but returns `None`.

- [ ] **Step 1: Write the failing test**

`tests/test_state_machine.py`:

```python
from babymon.config import EventRuleConfig
from babymon.rules.state import EvidenceStateMachine


def machine(**overrides) -> EvidenceStateMachine:
    cfg = EventRuleConfig(
        enter_threshold=0.5,
        exit_threshold=0.2,
        min_duration_s=2.0,
        exit_duration_s=3.0,
        cooldown_s=0.0,
        **overrides,
    )
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state_machine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.rules.state'`

- [ ] **Step 3: Write minimal implementation**

`babymon/rules/state.py`:

```python
"""The debouncing primitive every rule is built from.

Four controls, all configurable per event:

* hysteresis      - separate enter and exit thresholds, so a borderline
                    signal cannot flap.
* minimum duration- evidence must hold for N seconds before any state change.
* exit duration   - and must stay low for M seconds before the state clears.
* cooldown        - one alert per episode; a re-entry inside the cooldown
                    changes state silently.

This is the layer that decides whether a phone buzzes, and it is where a
monitor lives or dies on false-alarm rate.
"""

from __future__ import annotations

from typing import Literal

from babymon.config import EventRuleConfig

Transition = Literal["entered", "exited"]


class EvidenceStateMachine:
    def __init__(self, name: str, config: EventRuleConfig) -> None:
        self.name = name
        self.config = config
        self.active = False
        self.entered_at: float | None = None
        self._above_since: float | None = None
        self._below_since: float | None = None
        self._last_alert_ts: float | None = None

    def update(self, ts: float, value: float) -> Transition | None:
        cfg = self.config
        if not self.active:
            self._below_since = None
            if value >= cfg.enter_threshold:
                if self._above_since is None:
                    self._above_since = ts
                if ts - self._above_since >= cfg.min_duration_s:
                    return self._enter(ts)
            else:
                self._above_since = None
            return None

        self._above_since = None
        if value <= cfg.exit_threshold:
            if self._below_since is None:
                self._below_since = ts
            if ts - self._below_since >= cfg.exit_duration_s:
                return self._exit()
        else:
            self._below_since = None
        return None

    def _enter(self, ts: float) -> Transition | None:
        started = self._above_since if self._above_since is not None else ts
        self.active = True
        self.entered_at = started
        self._above_since = None
        in_cooldown = (
            self._last_alert_ts is not None
            and ts - self._last_alert_ts < self.config.cooldown_s
        )
        if in_cooldown:
            return None
        self._last_alert_ts = ts
        return "entered"

    def _exit(self) -> Transition:
        self.active = False
        self.entered_at = None
        self._below_since = None
        return "exited"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state_machine.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add babymon/rules/state.py tests/test_state_machine.py
git commit -m "feat: add evidence state machine with hysteresis and cooldown"
```

---

### Task 9: Rule engine — awake, zone exit, adult suppression

**Files:**
- Create: `babymon/rules/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `Observation` types (Task 1), `RulesConfig` (Task 3), `Zone` (Task 5), `EvidenceStateMachine` (Task 8).
- Produces: `RuleEngine(config: RulesConfig, crib_zone: Zone | None = None)` with `.handle(obs) -> list[Alert]` and `.tick(now) -> list[Alert]` (Task 10 adds tick's body), `.adult_present` (bool), `.last_baby_seen` (float | None).

- [ ] **Step 1: Write the failing test**

`tests/test_engine.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.rules.engine'`

- [ ] **Step 3: Write minimal implementation**

`babymon/rules/engine.py`:

```python
"""The rule engine: observations in, alerts out.

Single-threaded by design. Fed timestamped observations in order, a replay of
a recorded session produces identical output - which is what makes threshold
tuning tractable and the golden replay tests meaningful.
"""

from __future__ import annotations

from babymon.config import RulesConfig
from babymon.events import (
    HEALTH_ALERTS,
    Alert,
    AlertType,
    MotionEnergy,
    Observation,
    PersonBox,
    Severity,
)
from babymon.rules.state import EvidenceStateMachine
from babymon.rules.zones import Zone

SEVERITY = {
    AlertType.AWAKE: Severity.INFO,
    AlertType.CRYING: Severity.INFO,
    AlertType.ZONE_EXIT: Severity.WARNING,
    AlertType.PRONE: Severity.CRITICAL,
    AlertType.FACE_COVERED: Severity.CRITICAL,
    AlertType.LOST_TRACK: Severity.CRITICAL,
    AlertType.DETECTOR_SILENT: Severity.WARNING,
    AlertType.SOURCE_LOST: Severity.CRITICAL,
    AlertType.CAMERA_MOVED: Severity.WARNING,
}


class RuleEngine:
    def __init__(
        self, config: RulesConfig, crib_zone: Zone | None = None
    ) -> None:
        self.config = config
        self.crib_zone = crib_zone
        self.last_baby_seen: float | None = None
        self.last_observation_ts: dict[str, float] = {}
        self._last_adult_seen: float | None = None
        self._adult_since: float | None = None
        self._adult_active = False

        self._awake = EvidenceStateMachine("awake", config.awake)
        self._zone_exit = EvidenceStateMachine("zone_exit", config.zone_exit)

    @property
    def adult_present(self) -> bool:
        return self._adult_active

    def _note_adult(self, ts: float) -> None:
        if self._adult_since is None:
            self._adult_since = ts
        self._last_adult_seen = ts
        if ts - self._adult_since >= self.config.adult_min_duration_s:
            self._adult_active = True

    def _refresh_adult(self, ts: float) -> None:
        """Adult presence decays with time, not with contrary evidence.

        A detector that sees nothing emits nothing, so the engine cannot
        distinguish "the adult left" from "the detector is idle" by
        observation alone. Presence therefore expires ``adult_hold_s`` after
        the last adult sighting. Erring short is deliberate: a stale
        suppression would silence real alerts.
        """
        if self._last_adult_seen is None:
            return
        if ts - self._last_adult_seen > self.config.adult_hold_s:
            self._adult_active = False
            self._adult_since = None

    def handle(self, obs: Observation) -> list[Alert]:
        self.last_observation_ts[obs.detector] = obs.ts
        self._refresh_adult(obs.ts)
        alerts: list[Alert] = []

        if isinstance(obs, PersonBox):
            alerts.extend(self._handle_person(obs))
        elif isinstance(obs, MotionEnergy):
            alerts.extend(self._handle_motion(obs))

        return [a for a in alerts if self._allowed(a)]

    def _handle_person(self, obs: PersonBox) -> list[Alert]:
        alerts: list[Alert] = []
        if obs.label == "adult":
            # Adult presence is a suppressor, never an alert in itself.
            self._note_adult(obs.ts)
            return alerts

        if obs.label != "baby":
            return alerts

        self.last_baby_seen = obs.ts

        outside = (
            self.crib_zone is not None
            and not self.crib_zone.contains(obs.center)
        )
        if self._zone_exit.update(obs.ts, 1.0 if outside else 0.0) == "entered":
            alerts.append(
                self._alert(
                    AlertType.ZONE_EXIT,
                    self._zone_exit.entered_at or obs.ts,
                    obs.confidence,
                    {"bbox": list(obs.bbox)},
                )
            )
        return alerts

    def _handle_motion(self, obs: MotionEnergy) -> list[Alert]:
        if self._awake.update(obs.ts, obs.value) == "entered":
            return [
                self._alert(
                    AlertType.AWAKE,
                    self._awake.entered_at or obs.ts,
                    obs.confidence,
                    {"motion_energy": obs.value},
                )
            ]
        return []

    def _alert(
        self,
        type_: AlertType,
        started_at: float,
        confidence: float,
        metadata: dict,
    ) -> Alert:
        return Alert(
            type=type_,
            severity=SEVERITY[type_],
            started_at=started_at,
            confidence=confidence,
            metadata=metadata,
        )

    def _allowed(self, alert: Alert) -> bool:
        """Adult presence suppresses ordinary alerts. It never suppresses a
        health alert: a monitor that has failed must be noisy about it."""
        if alert.type in HEALTH_ALERTS:
            return True
        return not self.adult_present

    def tick(self, now: float) -> list[Alert]:
        """Time-driven checks. Filled in by the lost-track/watchdog task."""
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_engine.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add babymon/rules/engine.py tests/test_engine.py
git commit -m "feat: add rule engine with awake, zone exit and adult suppression"
```

---

### Task 10: Lost track and detector watchdog

**Files:**
- Modify: `babymon/rules/engine.py` (replace the `tick` stub, extend `__init__`)
- Test: `tests/test_engine_health.py`

**Interfaces:**
- Consumes: everything from Task 9.
- Produces: `RuleEngine(config, crib_zone=None, watched_detectors=())`; a working `.tick(now) -> list[Alert]` emitting `LOST_TRACK` and `DETECTOR_SILENT`.

- [ ] **Step 1: Write the failing test**

`tests/test_engine_health.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_engine_health.py -v`
Expected: FAIL — `RuleEngine() got an unexpected keyword argument 'watched_detectors'`

- [ ] **Step 3: Extend `__init__` in `babymon/rules/engine.py`**

Replace the `__init__` signature and add the health-tracking fields:

```python
    def __init__(
        self,
        config: RulesConfig,
        crib_zone: Zone | None = None,
        watched_detectors: tuple[str, ...] = (),
    ) -> None:
        self.config = config
        self.crib_zone = crib_zone
        self.watched_detectors = watched_detectors
        self.last_baby_seen: float | None = None
        self.last_observation_ts: dict[str, float] = {}
        self._last_adult_seen: float | None = None
        self._adult_since: float | None = None
        self._adult_active = False
        self._started_at: float | None = None
        self._lost_track_reported = False
        self._silent_reported: set[str] = set()

        self._awake = EvidenceStateMachine("awake", config.awake)
        self._zone_exit = EvidenceStateMachine("zone_exit", config.zone_exit)
```

- [ ] **Step 4: Clear the lost-track latch when the baby is seen**

In `_handle_person`, immediately after `self.last_baby_seen = obs.ts`, add:

```python
        self._lost_track_reported = False
```

- [ ] **Step 5: Replace the `tick` stub with the real implementation**

```python
    def tick(self, now: float) -> list[Alert]:
        """Time-driven checks.

        Absence of evidence is the failure mode that matters most: if the baby
        cannot be located, or a detector has gone quiet, nothing else in the
        system will fire and a naive monitor would report calm.
        """
        if self._started_at is None:
            self._started_at = now
        self._refresh_adult(now)
        alerts: list[Alert] = []
        alerts.extend(self._check_lost_track(now))
        alerts.extend(self._check_silent_detectors(now))
        return alerts

    def _check_lost_track(self, now: float) -> list[Alert]:
        if self._lost_track_reported:
            return []
        reference = (
            self.last_baby_seen
            if self.last_baby_seen is not None
            else self._started_at
        )
        if reference is None or now - reference < self.config.lost_track_after_s:
            return []
        self._lost_track_reported = True
        return [
            self._alert(
                AlertType.LOST_TRACK,
                reference,
                1.0,
                {"seconds_since_last_seen": round(now - reference, 1)},
            )
        ]

    def _check_silent_detectors(self, now: float) -> list[Alert]:
        alerts: list[Alert] = []
        for detector in self.watched_detectors:
            last = self.last_observation_ts.get(detector, self._started_at)
            silent = last is None or (
                now - last >= self.config.detector_silent_after_s
            )
            if silent and detector not in self._silent_reported:
                self._silent_reported.add(detector)
                alerts.append(
                    self._alert(
                        AlertType.DETECTOR_SILENT,
                        last if last is not None else now,
                        1.0,
                        {"detector": detector},
                    )
                )
            elif not silent:
                self._silent_reported.discard(detector)
        return alerts
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_engine_health.py tests/test_engine.py -v`
Expected: PASS, 14 tests. Both files must pass — Task 9's behaviour must not regress.

- [ ] **Step 7: Commit**

```bash
git add babymon/rules/engine.py tests/test_engine_health.py
git commit -m "feat: alert on lost track and silent detectors"
```

---

### Task 11: SQLite store sink with snapshots and retention

**Files:**
- Create: `babymon/sinks/__init__.py`, `babymon/sinks/base.py`, `babymon/sinks/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Alert` (Task 1).
- Produces: `Sink` protocol with `emit(alert, snapshot=None)`; `SqliteStore(db_path, snapshot_dir, retention_days=7, now=time.time)` with `.emit(alert, snapshot) -> int` (row id), `.recent(limit=50) -> list[dict]`, `.sweep() -> int`, `.close()`.

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.sinks'`

- [ ] **Step 3: Write minimal implementation**

`babymon/sinks/__init__.py` — empty file.

`babymon/sinks/base.py`:

```python
"""Sink protocol: where alerts go once the rule engine has decided."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from babymon.events import Alert


class Sink(Protocol):
    def emit(self, alert: Alert, snapshot: np.ndarray | None = None) -> object: ...

    def close(self) -> None: ...
```

`babymon/sinks/store.py`:

```python
"""SQLite event store with snapshot files and a retention sweeper.

An always-on camera will fill a disk, so retention is not optional.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from babymon.events import Alert

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
            name = f"{int(recorded_at * 1000)}_{alert.type.value}.jpg"
            path = self.snapshot_dir / name
            cv2.imwrite(str(path), snapshot)
            snapshot_path = str(path)

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
                json.dumps(alert.metadata),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add babymon/sinks tests/test_store.py
git commit -m "feat: add SQLite event store with snapshots and retention"
```

---

### Task 12: Runner, detector isolation, and CLI

**Files:**
- Create: `babymon/runner.py`, `babymon/__main__.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `DetectorWorker(detector, buffer, bus, stride=1, max_failures=5, stop_event=None)` with `.failures`, `.healthy`, `.run_once() -> bool`; `Pipeline(source, detectors, engine, sinks, strides=None, tick_interval_s=1.0, now=time.monotonic)` with `.run()`, `.stop()`, `.buffer`, `.bus`, `.workers`; `run_replay(source, detectors, engine, sinks) -> list[Alert]` for deterministic offline processing.

- [ ] **Step 1: Write the failing test**

`tests/test_runner.py`:

```python
import numpy as np

from babymon.bus import EventBus
from babymon.config import EventRuleConfig, RulesConfig
from babymon.events import AlertType, Frame, MotionEnergy
from babymon.rules.engine import RuleEngine
from babymon.runner import DetectorWorker, run_replay
from babymon.sources.base import FrameBuffer


class ExplodingDetector:
    name = "boom"

    def __init__(self):
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        raise RuntimeError("model blew up")


class CountingDetector:
    name = "counter"

    def __init__(self):
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        return [
            MotionEnergy(
                ts=frame.ts, detector=self.name, confidence=1.0, value=0.9
            )
        ]


class ListSink:
    def __init__(self):
        self.alerts = []

    def emit(self, alert, snapshot=None):
        self.alerts.append(alert)
        return len(self.alerts)

    def close(self):
        pass


def frame(seq: int) -> Frame:
    return Frame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        ts=float(seq),
        seq=seq,
        source_id="t",
    )


def test_worker_publishes_observations():
    buf, bus = FrameBuffer(), EventBus()
    sub = bus.subscribe()
    worker = DetectorWorker(CountingDetector(), buf, bus)
    buf.put(frame(0))
    assert worker.run_once() is True
    assert sub.get(timeout=0.1).value == 0.9


def test_a_failing_detector_does_not_raise():
    buf, bus = FrameBuffer(), EventBus()
    worker = DetectorWorker(ExplodingDetector(), buf, bus)
    buf.put(frame(0))
    assert worker.run_once() is True
    assert worker.failures == 1
    assert worker.healthy


def test_repeated_failures_mark_the_detector_unhealthy():
    buf, bus = FrameBuffer(), EventBus()
    worker = DetectorWorker(ExplodingDetector(), buf, bus, max_failures=3)
    for seq in range(3):
        buf.put(frame(seq))
        worker.run_once()
    assert worker.failures == 3
    assert not worker.healthy


def test_stride_skips_frames():
    det = CountingDetector()
    buf, bus = FrameBuffer(), EventBus()
    worker = DetectorWorker(det, buf, bus, stride=2)
    for seq in range(4):
        buf.put(frame(seq))
        worker.run_once()
    assert det.calls == 2  # seq 0 and 2


def test_replay_is_deterministic_and_reaches_the_sink():
    class StubSource:
        source_id = "t"

        def frames(self):
            for seq in range(6):
                yield frame(seq)

        def close(self):
            pass

    def make_engine():
        return RuleEngine(
            config=RulesConfig(
                awake=EventRuleConfig(
                    enter_threshold=0.5, exit_threshold=0.1,
                    min_duration_s=2.0, exit_duration_s=3.0, cooldown_s=0.0,
                ),
                lost_track_after_s=1e9,
            )
        )

    sink_a, sink_b = ListSink(), ListSink()
    a = run_replay(StubSource(), [CountingDetector()], make_engine(), [sink_a])
    b = run_replay(StubSource(), [CountingDetector()], make_engine(), [sink_b])

    assert [x.type for x in a] == [AlertType.AWAKE]
    assert [(x.type, x.started_at) for x in a] == [
        (x.type, x.started_at) for x in b
    ]
    assert sink_a.alerts == a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'babymon.runner'`

- [ ] **Step 3: Write the runner**

`babymon/runner.py`:

```python
"""Thread wiring and lifecycle.

A detector that raises is caught, counted, and isolated. It never takes down
the process - but it also never fails silently: once it exceeds its failure
budget it is marked unhealthy, stops producing, and the engine's watchdog
turns that silence into an alert.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable, Sequence

import numpy as np

from babymon.bus import EventBus, ReorderBuffer
from babymon.detectors.base import Detector
from babymon.events import Alert, Frame
from babymon.rules.engine import RuleEngine
from babymon.sinks.base import Sink
from babymon.sources.base import FrameBuffer, VideoSource

log = logging.getLogger(__name__)


class DetectorWorker:
    def __init__(
        self,
        detector: Detector,
        buffer: FrameBuffer,
        bus: EventBus,
        stride: int = 1,
        max_failures: int = 5,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.detector = detector
        self.buffer = buffer
        self.bus = bus
        self.stride = max(1, stride)
        self.max_failures = max_failures
        self.stop_event = stop_event or threading.Event()
        self.failures = 0
        self.healthy = True
        self._last_seq: int | None = None

    def run_once(self, timeout: float = 0.5) -> bool:
        """Process at most one frame. Returns True if a frame was consumed."""
        frame = self.buffer.get_latest(self._last_seq, timeout=timeout)
        if frame is None:
            return False
        self._last_seq = frame.seq
        if frame.seq % self.stride != 0:
            return True
        try:
            for obs in self.detector.process(frame):
                self.bus.publish(obs)
            self.failures = 0
        except Exception:
            self.failures += 1
            log.exception(
                "detector %s failed (%d/%d)",
                self.detector.name, self.failures, self.max_failures,
            )
            if self.failures >= self.max_failures:
                self.healthy = False
                log.error(
                    "detector %s disabled after repeated failures",
                    self.detector.name,
                )
        return True

    def run(self) -> None:
        while not self.stop_event.is_set() and self.healthy:
            self.run_once()


class Pipeline:
    def __init__(
        self,
        source: VideoSource,
        detectors: Sequence[Detector],
        engine: RuleEngine,
        sinks: Sequence[Sink],
        strides: dict[str, int] | None = None,
        tick_interval_s: float = 1.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.engine = engine
        self.sinks = list(sinks)
        self.now = now
        self.tick_interval_s = tick_interval_s
        self.buffer = FrameBuffer()
        self.bus = EventBus()
        self.stop_event = threading.Event()
        strides = strides or {}
        self.workers = [
            DetectorWorker(
                d, self.buffer, self.bus,
                stride=strides.get(d.name, 1),
                stop_event=self.stop_event,
            )
            for d in detectors
        ]
        self._latest_image: np.ndarray | None = None
        self._threads: list[threading.Thread] = []

    def _capture(self) -> None:
        try:
            for frame in self.source.frames():
                if self.stop_event.is_set():
                    return
                self._latest_image = frame.image
                self.buffer.put(frame)
        finally:
            self.stop_event.set()

    def _consume(self) -> None:
        sub = self.bus.subscribe()
        reorder = ReorderBuffer(self.engine.config.reorder_window_s)
        next_tick = self.now()
        while not self.stop_event.is_set():
            obs = sub.get(timeout=0.2)
            if obs is not None:
                reorder.add(obs)
            now = self.now()
            for ready in reorder.drain(now):
                self._dispatch(self.engine.handle(ready))
            if now >= next_tick:
                self._dispatch(self.engine.tick(now))
                next_tick = now + self.tick_interval_s

    def _dispatch(self, alerts: Iterable[Alert]) -> None:
        for alert in alerts:
            log.info("ALERT %s at %.1f", alert.type.value, alert.started_at)
            for sink in self.sinks:
                try:
                    sink.emit(alert, self._latest_image)
                except Exception:
                    log.exception("sink %s failed", type(sink).__name__)

    def run(self) -> None:
        self._threads = [
            threading.Thread(target=self._capture, name="capture", daemon=True),
            threading.Thread(target=self._consume, name="rules", daemon=True),
            *[
                threading.Thread(target=w.run, name=w.detector.name, daemon=True)
                for w in self.workers
            ],
        ]
        for t in self._threads:
            t.start()
        try:
            while not self.stop_event.wait(timeout=0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()
        self.source.close()
        for t in self._threads:
            t.join(timeout=2.0)
        for sink in self.sinks:
            sink.close()


def run_replay(
    source: VideoSource,
    detectors: Sequence[Detector],
    engine: RuleEngine,
    sinks: Sequence[Sink],
) -> list[Alert]:
    """Process a source synchronously, in frame order.

    No threads and no wall clock, so the same input always produces the same
    alert timeline. This is what golden replay tests and threshold tuning run
    against.
    """
    alerts: list[Alert] = []
    for frame in source.frames():
        observations = []
        for detector in detectors:
            try:
                observations.extend(detector.process(frame))
            except Exception:
                log.exception("detector %s failed during replay", detector.name)
        for obs in sorted(observations, key=lambda o: o.ts):
            alerts.extend(engine.handle(obs))
        alerts.extend(engine.tick(frame.ts))
    for alert in alerts:
        for sink in sinks:
            sink.emit(alert, None)
    return alerts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Write the CLI**

`babymon/__main__.py`:

```python
"""Command line entry point.

    python -m babymon --config config.yaml
    python -m babymon --config config.yaml --source clip.avi --replay
"""

from __future__ import annotations

import argparse
import logging
import sys

from babymon.config import Config
from babymon.detectors.motion import MotionDetector
from babymon.detectors.person import PersonDetector, YoloPersonModel
from babymon.rules.engine import RuleEngine
from babymon.rules.zones import Zone
from babymon.runner import Pipeline, run_replay
from babymon.sinks.store import SqliteStore
from babymon.sources.file import FileVideoSource
from babymon.sources.webcam import WebcamSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="babymon")
    p.add_argument("--config", help="path to YAML config file")
    p.add_argument("--source", help="video file to use instead of the webcam")
    p.add_argument(
        "--replay",
        action="store_true",
        help="process the file as fast as possible and exit",
    )
    p.add_argument("--no-person", action="store_true", help="skip YOLO detection")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config.load(args.config) if args.config else Config()
    if args.source:
        cfg.source.kind = "file"
        cfg.source.path = args.source

    crib_cfg = cfg.zone("crib")
    crib = Zone.from_config(crib_cfg) if crib_cfg else None
    if crib is None:
        logging.warning(
            "no 'crib' zone configured: zone exit and baby/adult attribution "
            "are disabled"
        )

    detectors = [
        MotionDetector(
            zone=crib, pixel_threshold=cfg.detectors.motion_pixel_threshold
        )
    ]
    if not args.no_person:
        detectors.append(
            PersonDetector(
                model=YoloPersonModel(cfg.detectors.model_path),
                crib_zone=crib,
                conf_threshold=cfg.detectors.person_conf_threshold,
                baby_max_area=cfg.detectors.baby_max_area,
            )
        )

    engine = RuleEngine(
        config=cfg.rules,
        crib_zone=crib,
        watched_detectors=tuple(d.name for d in detectors),
    )
    store = SqliteStore(
        db_path=cfg.storage.db_path,
        snapshot_dir=cfg.storage.snapshot_dir,
        retention_days=cfg.storage.retention_days,
    )
    store.sweep()

    if cfg.source.kind == "file":
        source = FileVideoSource(
            cfg.source.path,
            source_id=cfg.source.source_id,
            realtime=not args.replay,
        )
    else:
        source = WebcamSource(
            device=cfg.source.device, source_id=cfg.source.source_id
        )

    if args.replay:
        alerts = run_replay(source, detectors, engine, [store])
        for alert in alerts:
            print(f"{alert.started_at:8.2f}s  {alert.severity.value:8}  {alert.type.value}")
        store.close()
        return 0

    Pipeline(
        source=source,
        detectors=detectors,
        engine=engine,
        sinks=[store],
        strides={
            "motion": cfg.detectors.motion_stride,
            "person": cfg.detectors.person_stride,
        },
    ).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Verify the CLI starts**

Run: `python -m babymon --help`
Expected: argparse usage text listing `--config`, `--source`, `--replay`, `--no-person`, `--verbose`.

- [ ] **Step 7: Commit**

```bash
git add babymon/runner.py babymon/__main__.py tests/test_runner.py
git commit -m "feat: add pipeline runner, detector isolation and CLI"
```

---

### Task 13: Golden replay test and README

**Files:**
- Create: `tests/test_golden_replay.py`
- Modify: `README.md`
- Test: `tests/test_golden_replay.py`

**Interfaces:**
- Consumes: `run_replay` (Task 12), `FileVideoSource` (Task 4), `MotionDetector` (Task 6), `RuleEngine` (Tasks 9–10).
- Produces: a `synthetic_session` fixture and an end-to-end regression test asserting the alert timeline within a tolerance.

- [ ] **Step 1: Write the failing test**

`tests/test_golden_replay.py`:

```python
"""End-to-end regression net.

A synthetic session stands in for real footage until recordings exist: 6s
still, 4s of a moving block inside the crib, 6s still again. The expected
timeline is hand-derived from the configured thresholds, so a change that
silently shifts alert timing fails here.
"""

import numpy as np
import pytest

from babymon.config import EventRuleConfig, RulesConfig
from babymon.detectors.motion import MotionDetector
from babymon.events import AlertType
from babymon.rules.engine import RuleEngine
from babymon.rules.zones import Zone
from babymon.runner import run_replay
from babymon.sources.file import FileVideoSource
from tests.conftest import write_video

FPS = 10
CRIB = Zone(
    name="crib", polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
)


@pytest.fixture
def synthetic_session(tmp_path):
    frames = []
    for i in range(16 * FPS):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        second = i / FPS
        if 6.0 <= second < 10.0:
            # A block oscillating inside the crib zone. It must move on EVERY
            # frame: a single still frame drops motion energy to zero, which
            # resets the state machine's sustain clock and would stop the
            # alert from ever firing.
            offset = 10 + (i % 2) * 30
            img[100 : 100 + 40, offset + 120 : offset + 160] = 255
        frames.append(img)
    return write_video(tmp_path / "session.avi", frames, fps=FPS)


def build_engine() -> RuleEngine:
    return RuleEngine(
        config=RulesConfig(
            awake=EventRuleConfig(
                enter_threshold=0.01,
                exit_threshold=0.001,
                min_duration_s=1.0,
                exit_duration_s=2.0,
                cooldown_s=0.0,
            ),
            lost_track_after_s=1e9,       # no person detector in this test
            detector_silent_after_s=1e9,
        ),
        crib_zone=CRIB,
    )


def test_replay_produces_the_expected_alert_timeline(synthetic_session):
    alerts = run_replay(
        FileVideoSource(synthetic_session, realtime=False),
        [MotionDetector(zone=CRIB, pixel_threshold=25)],
        build_engine(),
        [],
    )
    assert [a.type for a in alerts] == [AlertType.AWAKE]
    # Motion starts at 6.0s; min_duration_s is 1.0s, so the episode is
    # credited from ~6.0s. Allow a 1s tolerance for encoder artefacts.
    assert alerts[0].started_at == pytest.approx(6.0, abs=1.0)


def test_replay_is_byte_for_byte_repeatable(synthetic_session):
    def once():
        return [
            (a.type, round(a.started_at, 3))
            for a in run_replay(
                FileVideoSource(synthetic_session, realtime=False),
                [MotionDetector(zone=CRIB, pixel_threshold=25)],
                build_engine(),
                [],
            )
        ]

    assert once() == once()


def test_a_still_session_produces_no_alerts(still_video):
    alerts = run_replay(
        FileVideoSource(still_video, realtime=False),
        [MotionDetector(zone=CRIB, pixel_threshold=25)],
        build_engine(),
        [],
    )
    assert alerts == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_golden_replay.py -v`
Expected: FAIL with `ImportError: cannot import name 'write_video' from 'tests.conftest'` if `tests/__init__.py` is missing, or an assertion failure. Fix by ensuring `tests/__init__.py` exists (created in Task 1).

- [ ] **Step 3: Make the test pass**

No new production code should be needed — this test exercises Tasks 4, 6, 9, 10 and 12 together. If it fails on timing, adjust only the tolerance in the test, and **only after confirming the alert is genuinely correct**; do not loosen a threshold in `babymon/` to make a test pass.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS, all tests across all files.

- [ ] **Step 5: Update the README**

Replace `README.md` with:

```markdown
# Baby AI Monitor

A local, privacy-preserving baby monitor. It watches a camera feed and raises
real-time safety alerts. All inference runs on your machine; no video ever
leaves it.

> **This is not a medical device.** It is an assistive monitor and nothing
> more. It must not be relied on for infant safety, and it does not detect or
> prevent SIDS. Follow safe-sleep guidance; this is a convenience layer on
> top of it.

## Status

Core pipeline and rule engine. Detects: baby awake (motion), baby out of the
crib zone, adult present (suppresses other alerts), and — critically — loss of
track and silent detectors.

Pose-based prone/face-covered detection, cry detection, and the web dashboard
are planned. See `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,detect]"
```

## Run

```bash
cp config.example.yaml config.yaml   # then edit the crib zone
python -m babymon --config config.yaml

# Replay a recording as fast as possible and print the alert timeline:
python -m babymon --config config.yaml --source clip.mp4 --replay
```

## Tuning

Every threshold in `config.example.yaml` is a starting point, not a tuned
value. Record a real session, run it through `--replay`, and adjust until the
alert timeline matches what you would have wanted. Thresholds that have not
been tuned against your own camera and room should not be trusted.

## Tests

```bash
python -m pytest
```
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_golden_replay.py README.md
git commit -m "test: add golden replay regression test; document usage"
```

---

## Verification

After Task 13, the following must all hold:

- [ ] `python -m pytest` passes with no failures and no skips.
- [ ] `python -m babymon --help` prints usage.
- [ ] `python -m babymon --source <a real clip> --replay --no-person` prints an alert timeline and writes rows into `data/events.db`.
- [ ] Killing the source mid-run (unplug the webcam) produces a `detector_silent` or `lost_track` alert rather than silence.

## Deferred to later plans

- **Plan 2:** pose detector (`PRONE`, `FACE_COVERED`), audio capture and cry detection, `CAMERA_MOVED` scene-geometry check.
- **Plan 3:** FastAPI dashboard, MJPEG preview with overlays, SSE timeline, canvas zone editor, ntfy push sink, clip recording with pre-roll.
- **Tuning:** all thresholds in `config.example.yaml` are untuned placeholders. Real footage is needed before any of them is trusted, and the spec's §9 low-light spike should run before the pose detector is built.
