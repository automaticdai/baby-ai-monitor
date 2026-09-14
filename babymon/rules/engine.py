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


def _started_at(machine: EvidenceStateMachine, fallback: float) -> float:
    """When the episode began, not when the transition fired.

    Explicitly `is not None`: ``entered_at or fallback`` silently discards a
    legitimate 0.0, which is exactly the case every replay of a recording
    starts with.
    """
    return machine.entered_at if machine.entered_at is not None else fallback


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
        self._lost_track_reported = False

        outside = (
            self.crib_zone is not None
            and not self.crib_zone.contains(obs.center)
        )
        if self._zone_exit.update(obs.ts, 1.0 if outside else 0.0) == "entered":
            alerts.append(
                self._alert(
                    AlertType.ZONE_EXIT,
                    _started_at(self._zone_exit, obs.ts),
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
                    _started_at(self._awake, obs.ts),
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
        # Through _allowed, exactly as handle() does. Health alerts pass
        # regardless of adult presence, but the engine must have ONE
        # suppression choke point, not two exits with one of them relying on
        # every future tick alert happening to be a health type.
        return [a for a in alerts if self._allowed(a)]

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
