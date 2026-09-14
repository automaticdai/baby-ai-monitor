"""Sink protocol: where alerts go once the rule engine has decided."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from babymon.events import Alert


class Sink(Protocol):
    def emit(self, alert: Alert, snapshot: np.ndarray | None = None) -> object: ...

    def close(self) -> None: ...
