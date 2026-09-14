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
