"""Detect optional third-party ``flybrain`` without loading MaleCNS files."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def flybrain_available() -> bool:
    """True if the third-party ``flybrain`` package is importable."""
    return importlib.util.find_spec("flybrain") is not None


def flybrain_data_available(data: Path | str | None = None) -> bool:
    """True if MaleCNS weight files are already on disk. Never downloads."""
    if not flybrain_available():
        return False
    from flybrain.data import DATA, has_data

    path = DATA if data is None else Path(data)
    return bool(has_data(path))


def require_flybrain() -> None:
    """Raise ImportError with the extra install hint if flybrain is missing."""
    if flybrain_available():
        return
    raise ImportError(
        "flybrain is an optional Model extra, not part of the fly-harness kernel. "
        "Install with: pip install 'fly-harness[flybrain]'. "
        "This extra docks a running third-party sim; it does not download MaleCNS "
        "unless you run `flybrain download` yourself."
    )
