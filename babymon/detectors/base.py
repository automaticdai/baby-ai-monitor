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
