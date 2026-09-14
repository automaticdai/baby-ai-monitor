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

## Known limitations

**A caregiver standing back from the cot may not suppress alerts.** One
threshold, `baby_max_area`, answers two different questions: "small enough to
be the baby" and "large enough to be an adult". A caregiver who occupies less
than that fraction of the frame, standing away from the cot, is reported as
`unknown` rather than `adult`, so suppression does not engage and an `awake`
alert can fire during a feed.

The obvious fix — a second `adult_min_area` threshold, so anything above it
counts as an adult wherever it stands — is **not** safe, and is deliberately
not implemented. Outside the crib, size alone cannot separate a distant adult
from a baby close to the camera, and guessing "adult" activates suppression:
the monitor would go quiet at the moment a baby had climbed out. The current
behaviour errs the other way, toward a spurious `info` alert. Separating these
two cases needs identity tracking across frames, and belongs with the pose
work rather than with another threshold.

**A lost-track alert during a feed is a `warning`, not a `critical`, only if
the caregiver is large enough in frame to register as an adult.** Same root
cause. A distant caregiver holding the baby still produces a `critical` after
`lost_track_after_s`. Raise `lost_track_after_s` above a typical feed if that
proves noisy, and tune once you have recordings of your own room.

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
