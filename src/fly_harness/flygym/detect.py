"""Detect an optional third-party FlyGym install without importing MuJoCo."""

from __future__ import annotations

import importlib.util


def flygym_available() -> bool:
    """True if the third-party ``flygym`` package is importable."""
    return importlib.util.find_spec("flygym") is not None


def require_flygym() -> None:
    """Raise ImportError with the extra install hint if FlyGym is missing."""
    if flygym_available():
        return
    raise ImportError(
        "FlyGym is an optional body extra, not part of the fly-harness core. "
        "Install with: pip install 'fly-harness[flygym]'"
    )
