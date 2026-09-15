"""Detect an optional third-party FlyGym install without importing MuJoCo."""

from __future__ import annotations

import importlib.util
from types import ModuleType


def flygym_available() -> bool:
    """True if ``flygym`` (1.x/2.x) or ``flygym_gymnasium`` is importable."""
    return (
        importlib.util.find_spec("flygym") is not None
        or importlib.util.find_spec("flygym_gymnasium") is not None
    )


def require_flygym() -> None:
    """Raise ImportError with the extra install hint if FlyGym is missing."""
    if flygym_available():
        return
    raise ImportError(
        "FlyGym is an optional body extra, not part of the fly-harness core. "
        "Install with: pip install 'fly-harness[flygym]'"
    )


def import_flygym_module() -> ModuleType:
    """Import FlyGym 1.x/2.x (``flygym``) or the Gymnasium fork."""
    require_flygym()
    for name in ("flygym", "flygym_gymnasium"):
        spec = importlib.util.find_spec(name)
        if spec is not None:
            return importlib.import_module(name)
    raise ImportError("FlyGym package disappeared after detection")
