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
        "flybrain is a third-party Model, not part of the fly-harness kernel. "
        "The harness does not dock a specific model. Install with: "
        "pip install 'fly-harness[flybrain]' to use it from the example or so "
        "biorouter can list model id 'flybrain.malecns'. "
        "This extra will not download MaleCNS; run `flybrain download` yourself."
    )
