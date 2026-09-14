# Baby AI Monitor — Design

Date: 2026-09-14
Status: Approved, ready for implementation planning

## 1. Purpose

A local, privacy-preserving baby monitor that watches a video and audio feed and
raises **real-time safety alerts** on a phone and a local dashboard.

The measure of success for v1 is alert quality, not feature count: alerts arrive
within seconds of the condition, false alarms are rare enough that the system
stays trusted, and the system is loud when it can no longer see.

**This is an assistive monitor, not a medical device.** It must never be
described as detecting, preventing, or reducing the risk of SIDS. Safe-sleep
practice is the intervention; this is a convenience layer above it. This
statement belongs in the README and in the dashboard's about panel.

### Non-goals for v1

- Sleep analytics, trends, weekly reports.
- Cloud storage, remote access outside the LAN, multi-home or multi-user support.
- Any cloud inference. All models run locally.
- Deployment to embedded hardware. v1 targets a laptop; the video source is an
  abstraction so a Pi or RTSP camera can be added later without redesign.

## 2. Scope

### Detected events

| Event | Signal | Risk |
|---|---|---|
| `AWAKE` | Sustained motion after a still period | Low |
| `CRYING` | Audio classifier over rolling windows | Low |
| `PRONE` / `FACE_COVERED` | Pose keypoint visibility | **High — weakest detector** |
| `ZONE_EXIT` | Baby box leaves the crib polygon | Medium |
| `ADULT_PRESENT` | Person box outside crib zone or above size threshold | Suppressor, not an alert |
| `UNKNOWN` / lost track | No confident baby observation for N seconds | **High** |
| System health | Source disconnect, detector silent or failing | High |

### Platform

- Camera: laptop webcam and recorded video files.
- Audio: laptop microphone and recorded audio.
- Stack: Python end-to-end. OpenCV, ultralytics/MediaPipe, FastAPI, SQLite,
  plain HTML/JS frontend.
- All inference local; no network egress except the push notification sink.

## 3. Architecture

One process, staged pipeline, threads at stage boundaries. Chosen over a
multi-process broker (unnecessary operational cost on a single machine) and over
a single synchronous loop (breaks as soon as one detector is slower than the
frame interval, and has nowhere to put audio).

```
                   ┌──────────────┐
  webcam / file ──►│   Capture    │──► latest-wins frame buffer (depth 1–2)
                   └──────────────┘              │
                                                 ├──► person detector ──┐
                                                 ├──► pose detector   ──┤
                                                 └──► motion detector ──┤
                                                                        ├──► event bus
  microphone   ──► Audio capture ──► ring buffer ──► cry detector    ──┘        │
                                                                                ▼
                                                                        ┌───────────────┐
                                                                        │  Rule engine  │  (single thread,
                                                                        │ state machines│   timestamp order)
                                                                        └───────────────┘
                                                                                │ Alerts
                                                          ┌─────────────────────┼─────────────────┐
                                                          ▼                     ▼                 ▼
                                                    SQLite + snapshots       ntfy push       SSE → dashboard
```

### Threading model

- **Capture thread** (one per source). Owns the device, stamps each frame with a
  monotonic timestamp and sequence number, writes to a **latest-wins** buffer.
  Latest-wins rather than a queue: for real-time alerting a stale frame has no
  value, so under load the system drops old frames instead of accumulating
  backlog. Drop count is a health metric surfaced on the dashboard.
- **Detector worker threads** (one per detector). Each reads the current frame
  and runs at its own configured stride (person every frame, pose every 2nd–3rd,
  motion every frame). A slow detector degrades only its own rate. The heavy
  models release the GIL in native code, so this parallelises acceptably.
- **Rule engine thread** (exactly one). Consumes the bus in timestamp order.
  Single-threaded by design: all state transitions are deterministic and
  therefore replayable and testable.
- **Web thread** — FastAPI/uvicorn serving dashboard, MJPEG preview, SSE stream.

The event bus is an in-process pub/sub behind a narrow interface. It is the
designated seam: replacing it with ZeroMQ or Redis splits the pipeline across
machines without touching detectors or rules.

### Core types

Keeping these three distinct is the backbone of the design.

- **`Frame`** — image, monotonic timestamp, sequence number, source id.
- **`Observation`** — what one detector saw at one instant, always carrying a
  confidence. Stateless, no interpretation:
  `PersonBox(bbox, conf)`, `Pose(keypoints, visibility)`, `MotionEnergy(0..1)`,
  `CryProbability(p)`.
- **`Alert`** — a decision the rule engine reached over time, e.g. "prone for
  12s, no adult present". **Only Alerts reach the user.** No detector can notify
  anyone directly.

### Module layout

```
babymon/
  config.py         # YAML → pydantic models, validated at startup
  bus.py            # in-process pub/sub; the seam for later splitting
  events.py         # Frame / Observation / Alert types
  sources/
    base.py         # VideoSource + AudioSource protocols
    webcam.py
    file.py         # timing-accurate replay of recorded video
    audio.py
  detectors/
    base.py         # Detector protocol: process(...) -> list[Observation]
    person.py       # YOLO11n, person class
    pose.py         # MediaPipe Pose / YOLO11n-pose
    motion.py       # frame differencing within crib zone
    cry.py          # YAMNet over ~1s audio windows
  rules/
    engine.py       # state machines, hysteresis, suppression, watchdog
    zones.py        # ROI polygons, point-in-polygon, geometry checks
  sinks/
    base.py
    store.py        # SQLite writer, snapshot and clip files, retention sweeper
    ntfy.py
  web/
    app.py          # FastAPI: dashboard, MJPEG, SSE, config API
    static/
  runner.py         # wiring and lifecycle
  __main__.py
```

`sources/file.py` is load-bearing, not a convenience. A video-file source that
replays with correct inter-frame timing is what makes detector tuning and
end-to-end testing possible without a live baby in front of the camera.

## 4. Detectors

| Detector | Model | Emits | Default stride |
|---|---|---|---|
| `person` | YOLO11n, person class only | `PersonBox` | every frame |
| `pose` | MediaPipe Pose or YOLO11n-pose | `Pose` | every 3rd frame |
| `motion` | frame differencing in crib zone | `MotionEnergy` | every frame |
| `cry` | YAMNet, ~1s windows | `CryProbability` | own audio clock |

### Baby vs adult is a geometric heuristic, not a model

No off-the-shelf model classifies "infant". v1 discriminates by geometry: a
person box whose centre lies inside the crib zone and whose area is below a
configured threshold is the baby; larger boxes, or boxes centred outside the
zone, are an adult.

This holds for a fixed camera over a crib and fails if the camera moves. The
zone configuration therefore records the camera framing at setup time, and a
large sudden change in scene geometry raises a **"camera moved, zones may be
invalid"** warning rather than silently mis-attributing every subsequent
detection.

### Prone and face-covered come from keypoint visibility

Face keypoints (nose, eyes, ears) visible together with torso keypoints → face
up. Torso visible with face keypoints absent → likely prone or covered.

This is the weakest detector in the system and will need real footage to tune.
It is specified as visibility logic rather than a classifier so its failure mode
is legible: when it cannot see, it must degrade into `UNKNOWN` rather than into
an implicit "safe".

## 5. Rule engine

One state machine per event type. Each exposes the same four controls, all
configurable per event:

- **Hysteresis** — distinct enter and exit thresholds so a borderline signal
  cannot flap. E.g. enter `PRONE` above confidence 0.7, leave only below 0.4.
- **Minimum duration** — evidence must hold across a sliding window of N seconds
  before any state change. Eliminates single-frame noise.
- **Cooldown** — one alert per continuous episode; re-arms once the state
  clears.
- **Suppression** — `adult_present` gates every other alert, so leaning over the
  crib does not buzz your phone.

### `UNKNOWN` is not `SAFE`

If the baby cannot be located — blanket fully over, lights off, lens obstructed
— no detector fires, and a naive system reports calm. The engine tracks
time-since-last-confident-baby-observation and raises a distinct **lost track**
alert past a configurable threshold.

A monitor that goes quiet because it stopped seeing anything is the precise
failure mode that matters. This requirement generalises: every silent failure in
the system must produce noise.

### Determinism

Because the engine is single-threaded and consumes timestamped observations in
order, replaying a recorded session produces byte-identical alert output. This
property is what makes threshold tuning tractable and the test suite meaningful.

## 6. Storage, dashboard, delivery

**Storage.** SQLite events table:
`id, type, started_at, ended_at, severity, confidence, snapshot_path, clip_path,
metadata_json`. Snapshots and clips as files on disk. A rolling in-memory frame
ring buffer lets each alert save a clip with **pre-roll** — the seconds before
the event are the ones worth reviewing. Retention is configurable (default 7
days), enforced by a sweeper, because an always-on camera will otherwise fill
the disk.

**Dashboard** (FastAPI + plain HTML/JS):
- Live MJPEG preview with detection overlays.
- SSE-driven event timeline with snapshot thumbnails.
- Canvas zone editor; polygons write back to config.
- Health panel: per-detector FPS, frame drop count, model latency, source state.
- About panel carrying the not-a-medical-device statement.

**Delivery.** ntfy as the v1 push sink — no account required, self-hostable,
supports attached snapshots. Behind the same `Sink` interface as the store, so
Telegram or Pushover drop in without touching the engine.

**Configuration.** A single YAML file parsed into pydantic models and validated
at startup: source selection, zone polygons, per-event thresholds, retention,
sink credentials. Invalid config fails loudly at startup rather than at 3am.

## 7. Failure handling

Treated as a feature, not as error paths.

- **Source disconnect** → reconnect with exponential backoff; sustained failure
  raises an alert.
- **Detector exception** → caught and counted, marks that detector unhealthy,
  never takes down the process. Repeated failures disable the detector and
  alert.
- **Model load failure** → fatal at startup. Better to refuse to start than to
  run half-blind.
- **Watchdog** → verifies every detector is still producing observations. A
  silent detector raises an alert, for the same reason `UNKNOWN` is not `SAFE`.
- **Disk pressure** → retention sweeper; if writes still fail, alert rather than
  silently stop recording.

## 8. Testing

TDD throughout, in three layers.

1. **Rule engine tests** — synthetic observation streams fed at controlled
   timestamps, asserting exact alert output. No models, no video, millisecond
   runtime. The bulk of the suite lives here: hysteresis, minimum duration,
   cooldown, suppression, `UNKNOWN` transitions, watchdog firing.
2. **Detector tests** — fixed fixture images and audio clips, asserting
   observation shape and rough confidence bands. Deliberately tolerant, since
   model outputs drift between versions.
3. **Golden replay tests** — recorded clips with a hand-labelled expected alert
   timeline, run end-to-end through the file source. The regression net for
   false-alarm rate.

**Latency budget**, asserted in tests against the replay clips:
- Motion- and audio-triggered alerts: within **2s** of onset.
- Pose-based alerts (`PRONE`, `FACE_COVERED`): within **5s** of onset.

## 9. Open items for implementation planning

- Choice between MediaPipe Pose and YOLO11n-pose — decide by measuring both on
  real crib footage, including in low light and with IR-style greyscale input.
- Initial threshold values for every state machine are placeholders until tuned
  against recorded sessions; the plan should include a tuning step with real
  footage before any thresholds are treated as final.
- Low-light behaviour is unaddressed in v1 and is the most likely source of
  `UNKNOWN` states on a laptop webcam. Worth an early spike.
