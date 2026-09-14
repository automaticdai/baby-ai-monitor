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
