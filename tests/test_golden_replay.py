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
    # credited from ~6.0s. The pipeline is bit-exact (proved below by
    # test_replay_is_byte_for_byte_repeatable), so the tolerance here is
    # tight - about one frame at 10fps - not a wide margin for jitter.
    assert alerts[0].started_at == pytest.approx(6.0, abs=0.1)


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


@pytest.fixture
def motion_outside_crib_session(tmp_path):
    """Same shape as `synthetic_session`, but the moving block sits entirely
    in the far-left margin of the frame, well clear of the crib polygon.

    The crib zone (see `CRIB` above) rasterises to x in [64, 256], y in
    [48, 192] on a 240x320 frame (verified directly against `Zone.mask`).
    This block's x range is [0, 60] - always at least 4px left of the
    zone's left edge - so every pixel it ever touches is outside the mask.
    If `MotionDetector`'s zone restriction were ever bypassed, this motion
    would clear `enter_threshold` just as easily as the in-zone fixture
    does, and this test would catch it; with the mask applied, the
    detector must report zero energy and no alert should fire.
    """
    frames = []
    for i in range(16 * FPS):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        second = i / FPS
        if 6.0 <= second < 10.0:
            # Same "move on every frame" discipline as synthetic_session.
            offset = (i % 2) * 20
            img[100 : 100 + 40, offset : offset + 40] = 255
        frames.append(img)
    return write_video(tmp_path / "outside_crib.avi", frames, fps=FPS)


def test_motion_outside_the_crib_zone_produces_no_alert(
    motion_outside_crib_session,
):
    # Confirms, independently of run_replay, that the masked detector sees
    # zero motion energy throughout - not merely that no alert happened to
    # cross the (much lower) enter_threshold.
    detector = MotionDetector(zone=CRIB, pixel_threshold=25)
    values = [
        obs.value
        for frame in FileVideoSource(
            motion_outside_crib_session, realtime=False
        ).frames()
        for obs in detector.process(frame)
    ]
    assert max(values) == 0.0

    alerts = run_replay(
        FileVideoSource(motion_outside_crib_session, realtime=False),
        [MotionDetector(zone=CRIB, pixel_threshold=25)],
        build_engine(),
        [],
    )
    assert alerts == []


@pytest.fixture
def short_burst_session(tmp_path):
    """Motion inside the crib, moving every frame exactly as
    `synthetic_session` does, but sustained for only 0.4s - well under the
    configured `min_duration_s` of 1.0s used by `build_engine()`.

    This pins the threshold from below: `synthetic_session` (4s of motion)
    proves the alert fires once evidence is sustained long enough;
    without this fixture, `_started_at`'s back-dating to the first
    threshold crossing means any min_duration_s between 0 and ~3.9s would
    produce the identical started_at == 6.0, so the sustained test alone
    cannot tell 1.0s from an accidental 3.5s. This test would fail
    (wrongly alert) if min_duration_s were silently ignored or shrunk.
    """
    frames = []
    for i in range(16 * FPS):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        second = i / FPS
        if 6.0 <= second < 6.4:
            offset = 10 + (i % 2) * 30
            img[100 : 100 + 40, offset + 120 : offset + 160] = 255
        frames.append(img)
    return write_video(tmp_path / "short_burst.avi", frames, fps=FPS)


def test_a_burst_shorter_than_min_duration_does_not_alert(short_burst_session):
    # Confirm the burst genuinely crosses enter_threshold - the point is
    # that it crosses and then stops too soon, not that it never crosses.
    detector = MotionDetector(zone=CRIB, pixel_threshold=25)
    values = [
        obs.value
        for frame in FileVideoSource(short_burst_session, realtime=False).frames()
        for obs in detector.process(frame)
    ]
    assert max(values) > 0.01  # enter_threshold used by build_engine()

    alerts = run_replay(
        FileVideoSource(short_burst_session, realtime=False),
        [MotionDetector(zone=CRIB, pixel_threshold=25)],
        build_engine(),
        [],
    )
    assert alerts == []
