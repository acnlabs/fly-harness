"""MCP extra tests: handlers and mocked FastMCP — no live MCP client."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from fly_harness.mcp.detect import mcp_available, require_mcp
from fly_harness.mcp.handlers import HarnessSession, serialize_value
from fly_harness.mcp.server import create_mcp_server


class FakeFastMCP:
    """Stand-in for FastMCP: records tools, never opens stdio."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.tools: dict[str, Callable[..., Any]] = {}
        self.ran = False

    def tool(self, *args: Any, **kwargs: Any) -> Any:
        if args and callable(args[0]):
            fn = args[0]
            self.tools[fn.__name__] = fn
            return fn

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[fn.__name__] = fn
            return fn

        return deco

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.ran = True


def _core_sources() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "src" / "fly_harness"
    return [
        root / "__init__.py",
        root / "backend.py",
        root / "brain_state.py",
        root / "harness.py",
        root / "http_backend.py",
        root / "protocols.py",
        root / "connectome_loader.py",
    ]


def test_kernel_sources_do_not_import_mcp() -> None:
    for path in _core_sources():
        text = path.read_text(encoding="utf-8")
        assert "fastmcp" not in text
        assert "fly_harness.mcp" not in text
        assert "from mcp" not in text
        assert "import mcp" not in text


def test_handlers_import_without_fastmcp() -> None:
    session = HarnessSession()
    status = session.status()
    assert status["kernel"] == "FlyHarness.step"
    assert status["n_neurons"] == 24
    assert status["extension"] == "mcp"


def test_require_mcp_hints_extra_when_missing() -> None:
    if mcp_available():
        require_mcp()
        return
    with pytest.raises(ImportError, match="fly-harness\\[mcp\\]"):
        require_mcp()


def test_harness_step_handler_returns_jsonable_action() -> None:
    session = HarnessSession()
    idle = session.step(touch_left=0.0, touch_right=0.0)
    assert idle["n_neurons"] == 24
    assert idle["timestamp"] == 1.0
    assert set(idle["action"]) >= {"turn", "forward", "brake"}
    assert isinstance(idle["potentials"], list)
    assert len(idle["potentials"]) == 24


def test_harness_step_left_touch_biases_right_turn() -> None:
    session = HarnessSession()
    session.step()
    session.step(touch_left=1.0)
    result = session.step(touch_left=1.0)
    assert result["action"]["turn"] == 1


def test_harness_reset_zeros_clock() -> None:
    session = HarnessSession()
    session.step(touch_left=1.0)
    out = session.reset()
    assert out["timestamp"] == 0.0
    assert session.status()["timestamp"] == 0.0


def test_serialize_value_handles_mapping_and_numpy() -> None:
    import numpy as np

    payload = serialize_value({"x": np.float64(1.5), "y": np.arange(2)})
    assert payload == {"x": 1.5, "y": [0, 1]}


def test_create_server_with_fake_fastmcp_does_not_run() -> None:
    fake = FakeFastMCP
    mcp = create_mcp_server(mcp_factory=fake)
    assert isinstance(mcp, FakeFastMCP)
    assert not mcp.ran
    assert set(mcp.tools) == {"harness_step", "harness_reset", "harness_status"}
    stepped = mcp.tools["harness_step"](touch_left=0.0, touch_right=0.0)
    assert stepped["n_neurons"] == 24
    assert mcp.tools["harness_status"]()["kernel"] == "FlyHarness.step"
    mcp.tools["harness_reset"]()
    assert mcp.tools["harness_status"]()["timestamp"] == 0.0


@pytest.mark.skipif(not mcp_available(), reason="fastmcp not installed")
def test_fastmcp_factory_smoke_without_stdio() -> None:
    mcp = create_mcp_server()
    assert hasattr(mcp, "run")
    assert mcp is not None
