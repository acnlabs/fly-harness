"""FastMCP stdio server wrapping HarnessSession — imported only by the extra.

The kernel (BrainState, FlyHarness) does not import this module.
"""

from __future__ import annotations

from typing import Any, Callable

from fly_harness.mcp.detect import require_mcp
from fly_harness.mcp.handlers import HarnessSession

_INSTRUCTIONS = (
    "Optional MCP extension pack for fly-harness. Tools call FlyHarness.step on a "
    "sparse circuit (default: 24-neuron reflex fixture). This is not a 140k-neuron "
    "FlyWire brain, not FlyGym, and not a consciousness claim."
)


def create_mcp_server(
    session: HarnessSession | None = None,
    *,
    mcp_factory: Callable[..., Any] | None = None,
) -> Any:
    """Build a FastMCP (or injectable) server whose tools call HarnessSession.

    Pass ``mcp_factory`` in tests to avoid importing FastMCP or opening stdio.
    """
    if mcp_factory is None:
        require_mcp()
        from fastmcp import FastMCP

        mcp_factory = FastMCP

    bound = session if session is not None else HarnessSession()
    mcp = mcp_factory(name="fly-harness", instructions=_INSTRUCTIONS)

    @mcp.tool
    def harness_step(touch_left: float = 0.0, touch_right: float = 0.0) -> dict:
        """One FlyHarness.step: encode touch observation, advance dynamics, decode action."""
        return bound.step(touch_left=touch_left, touch_right=touch_right)

    @mcp.tool
    def harness_reset() -> dict:
        """Zero membrane potentials and the session clock."""
        return bound.reset()

    @mcp.tool
    def harness_status() -> dict:
        """Harness size, timestamp, and package version. Not a whole-brain dump."""
        return bound.status()

    return mcp


def main() -> None:
    """Run the MCP server over stdio (default FastMCP transport)."""
    require_mcp()
    create_mcp_server().run()


if __name__ == "__main__":
    main()
