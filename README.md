# Baby AI Monitor

A local, privacy-preserving baby monitor. It watches a camera feed and raises
real-time safety alerts. All inference runs on your machine; no video ever
leaves it.

> **This is not a medical device.** It is an assistive monitor and nothing
> more. It must not be relied on for infant safety, and it does not detect or
> prevent SIDS. Follow safe-sleep guidance; this is a convenience layer on
> top of it.

## Status

Core pipeline and rule engine. Detects: baby awake (motion), adult present
(suppresses other alerts), and — critically — loss of track and silent
detectors.

Zone-exit detection is not implemented. Telling "the baby climbed out" apart
from "a pet walked past" needs identity tracking across frames, which the
current single-frame geometry cannot do. A baby who leaves the crib therefore
stops being seen and surfaces as a lost-track alert rather than as a zone exit:
less precise, but never silent.

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
