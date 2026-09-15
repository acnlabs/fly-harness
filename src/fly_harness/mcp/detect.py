"""Detect the optional FastMCP extra without importing it into the kernel."""

from __future__ import annotations

import importlib.util


def mcp_available() -> bool:
    """True if the third-party ``fastmcp`` package is importable."""
    return importlib.util.find_spec("fastmcp") is not None


def require_mcp() -> None:
    """Raise ImportError with the extra install hint if FastMCP is missing."""
    if mcp_available():
        return
    raise ImportError(
        "MCP is an optional protocol extra, not part of the fly-harness kernel. "
        "Install with: pip install 'fly-harness[mcp]'"
    )
