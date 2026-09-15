"""Optional MCP protocol pack around FlyHarness.step.

Not the microkernel. Install with ``pip install 'fly-harness[mcp]'``.
Handlers import without FastMCP; the stdio server needs the extra.
"""

from fly_harness.mcp.detect import mcp_available, require_mcp
from fly_harness.mcp.handlers import HarnessSession, serialize_step_result
from fly_harness.mcp.server import create_mcp_server

__all__ = [
    "HarnessSession",
    "create_mcp_server",
    "mcp_available",
    "require_mcp",
    "serialize_step_result",
]
