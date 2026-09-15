"""Composed MCP + FlyGym example: no MuJoCo, no live MCP client."""

from __future__ import annotations

import importlib.util
from typing import Any, Callable

import numpy as np
import pytest

from fly_harness.demo.composed import (
    FIXTURE_N_NEURONS,
    BodyAttachedSession,
    StubFlyGymEnv,
    TouchOrBodyEncoder,
    build_composed_loop,
    main,
    observation_for_encoder,
    run_scripted_loop,
)
from fly_harness.flygym import N_LEG_JOINTS
from fly_harness.mcp.server import create_mcp_server


class FakeFastMCP:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
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


def test_composed_loop_stays_on_fixture_size() -> None:
    loop = build_composed_loop()
    assert loop.n_neurons == FIXTURE_N_NEURONS == 24
    assert loop.n_neurons != 166_000
    assert isinstance(loop.body.env, StubFlyGymEnv)


def test_touch_observation_becomes_flygym_contact() -> None:
    obs = observation_for_encoder({"touch_left": 1.0, "touch_right": 0.0})
    assert obs["contact_forces"].shape == (36, 3)
    assert float(obs["contact_forces"][0, 2]) == 1.0
    assert float(obs["contact_forces"][-1, 2]) == 0.0


def test_touch_or_body_encoder_maps_mcp_touch() -> None:
    encoder = TouchOrBodyEncoder(24)
    currents = encoder.encode({"touch_left": 1.0, "touch_right": 0.0})
    assert currents.shape == (24,)
    assert currents[0] > currents[2]


def test_session_step_applies_joints_to_stub_body() -> None:
    env = StubFlyGymEnv()
    loop = build_composed_loop(env=env)
    out = loop.session.step(touch_left=1.0, touch_right=0.0)
    assert out["n_neurons"] == 24
    assert out["fixture_n_neurons"] == 24
    assert env.actions, "body env should receive a FlyGym action dict"
    assert env.actions[0]["joints"].shape == (N_LEG_JOINTS,)
    assert env.actions[0]["adhesion"].shape == (6,)
    assert "joints" in out["body_action"]


def test_scripted_loop_is_mcp_shaped_and_finite() -> None:
    loop = build_composed_loop()
    traces = run_scripted_loop(loop, steps=3)
    assert len(traces) == 3
    assert traces[-1]["timestamp"] == 3.0
    assert all(row["n_neurons"] == 24 for row in traces)


def test_mcp_tool_calls_composed_step_without_stdio() -> None:
    env = StubFlyGymEnv()
    loop = build_composed_loop(env=env)
    assert isinstance(loop.session, BodyAttachedSession)
    mcp = create_mcp_server(loop.session, mcp_factory=FakeFastMCP)
    assert not mcp.ran
    result = mcp.tools["harness_step"](touch_left=1.0, touch_right=0.0)
    assert result["n_neurons"] == 24
    assert len(env.actions) == 1
    assert np.asarray(result["body_action"]["joints"]).shape == (N_LEG_JOINTS,)
    mcp.tools["harness_reset"]()
    assert loop.harness.state.timestamp == 0.0


def test_kernel_init_does_not_import_composed() -> None:
    from pathlib import Path

    init_text = (
        Path(__file__).resolve().parents[1] / "src" / "fly_harness" / "__init__.py"
    ).read_text(encoding="utf-8")
    assert "demo.composed" not in init_text
    assert "fastmcp" not in init_text


def test_main_scripted_mentions_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--steps", "2"])
    out = capsys.readouterr().out
    assert "24" in out
    assert "166k" in out
    assert "MaleCNS" in out


@pytest.mark.skipif(importlib.util.find_spec("fastmcp") is None, reason="fastmcp not installed")
def test_composed_fastmcp_factory_without_client() -> None:
    loop = build_composed_loop()
    mcp = create_mcp_server(loop.session)
    assert hasattr(mcp, "run")
