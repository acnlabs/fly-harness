"""Detect optional third-party ``c302`` without starting OpenWorm.

Not a fly-harness extra. Extra glue exists only when a third-party API is
not already a ``ModelBackend``; that path was rejected for worm. This
module never downloads connectomes and never starts OpenWorm Docker.
"""

from __future__ import annotations

import importlib.util
from typing import Any


def c302_available() -> bool:
    """True if third-party ``c302`` is already importable. Never installs it."""
    return importlib.util.find_spec("c302") is not None


def running_c302_backend() -> Any | None:
    """Return a tick/reset Model if the environment already has one.

    ``c302`` on its own is a NeuroML generator, not a running sim. This
    repo does not wrap NEURON or start OpenWorm Docker, so the default is
    ``None`` (caller skips / omits the id). Deploy a ``ModelBackend`` and
    list ``c302.celegans`` yourself.
    """
    if not c302_available():
        return None
    return None
