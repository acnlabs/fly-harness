"""Compose optional MCP + FlyGym extras through FlyHarness.step.

This path uses the 24-neuron reflex **fixture**. It does not load FlyWire,
MaleCNS (~166k neurons), or any dense whole-brain matrix. FlyGym and FastMCP
are not imported unless you ask the example to serve or to try a real sim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fly_harness.demo.connectome import N_NEURONS, SENSORY_INDICES, build_reflex_connectome
from fly_harness.flygym import (
    N_LEG_JOINTS,
    FlyGymDecoder,
    FlyGymEncoder,
    FlyGymHarnessEnv,
    apply_flygym_action,
    coerce_action,
)
from fly_harness.harness import FlyHarness
from fly_harness.mcp.handlers import HarnessSession, serialize_step_result

FIXTURE_N_NEURONS = N_NEURONS  # 24 — not 140k / 166k
_BANNER = (
    "fly-harness composed example: MCP protocol + FlyGym body through "
    f"FlyHarness.step on a {FIXTURE_N_NEURONS}-neuron fixture "
    "(not a 140k/166k CNS upload)."
)


class StubFlyGymEnv:
    """In-process NeuroMechFly stand-in so the example runs without MuJoCo."""

    def __init__(self) -> None:
        self.actions: list[dict[str, np.ndarray]] = []
        self.obs: dict[str, np.ndarray] = {
            "joints": np.zeros((3, N_LEG_JOINTS), dtype=np.float64),
            "contact_forces": np.zeros((36, 3), dtype=np.float64),
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return self.obs, {}

    def step(self, action: dict[str, np.ndarray]) -> tuple:
        self.actions.append(action)
        return self.obs, 0.0, False, False, {}


class TouchOrBodyEncoder(FlyGymEncoder):
    """FlyGymEncoder that also accepts MCP-style ``touch_left`` / ``touch_right`` dicts."""

    def encode(self, observation: Any) -> np.ndarray:
        return super().encode(observation_for_encoder(observation))


def observation_for_encoder(
    observation: Any,
    *,
    last_body_obs: Any | None = None,
) -> Any:
    """Merge MCP touch fields into a FlyGym-shaped observation dict."""
    payload: dict[str, Any]
    if isinstance(observation, Mapping):
        payload = dict(observation)
    else:
        return observation

    touch_left = payload.get("touch_left")
    touch_right = payload.get("touch_right")
    has_touch = touch_left is not None or touch_right is not None
    has_gym = "joints" in payload or "joint_angles" in payload or "contact_forces" in payload
    if has_gym and not has_touch:
        return payload

    base: dict[str, Any] = {}
    if isinstance(last_body_obs, Mapping):
        base.update(last_body_obs)
    if has_gym:
        base.update(payload)

    if has_touch:
        left = float(touch_left or 0.0)
        right = float(touch_right or 0.0)
        contact = np.zeros((36, 3), dtype=np.float64)
        contact[:18, 2] = left
        contact[18:, 2] = right
        base["contact_forces"] = contact
        base.setdefault("joints", np.zeros((3, N_LEG_JOINTS), dtype=np.float64))
        return base
    return payload if payload else observation


class BodyAttachedSession(HarnessSession):
    """MCP-callable session: FlyHarness.step then apply the action to a FlyGym-shaped env.

    Does not rewrite the MCP server or FlyGym. ``create_mcp_server(session)`` keeps
    the existing ``harness_step`` tool; this class is the composition.
    """

    def __init__(self, harness: FlyHarness, body: FlyGymHarnessEnv) -> None:
        super().__init__(harness)
        self.body = body

    def reset(self) -> dict[str, Any]:
        out = super().reset()
        self.body.reset()
        return out

    def step(
        self,
        observation: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = dict(observation or {})
        for key, value in fields.items():
            if value is not None and key not in payload:
                payload[key] = value
        encoder_obs = observation_for_encoder(
            payload, last_body_obs=getattr(self.body, "_last_obs", None)
        )
        result = self.harness.step(encoder_obs)
        action = coerce_action(result.action, rest_pose=self.body.rest_pose)
        raw = apply_flygym_action(
            self.body.env,
            action,
            fly_name=self.body.fly_name,
            actuator_type=self.body.actuator_type,
        )
        obs = _observation_from_env_step(raw)
        if obs is not None:
            self.body._last_obs = obs
        payload_out = serialize_step_result(result, n_neurons=self.harness.state.n_neurons)
        payload_out["body_action"] = action.as_env_dict()
        payload_out["fixture_n_neurons"] = FIXTURE_N_NEURONS
        return payload_out


@dataclass
class ComposedLoop:
    """Harness + FlyGym-shaped body + MCP session sharing one BrainState."""

    harness: FlyHarness
    body: FlyGymHarnessEnv
    session: BodyAttachedSession

    @property
    def n_neurons(self) -> int:
        return int(self.harness.state.n_neurons)


def _observation_from_env_step(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, tuple) and raw:
        return raw[0]
    return raw


def build_composed_harness() -> FlyHarness:
    state = build_reflex_connectome()
    if state.n_neurons != FIXTURE_N_NEURONS:
        raise RuntimeError("composed example must stay on the 24-neuron fixture")
    return FlyHarness(
        state,
        TouchOrBodyEncoder(state.n_neurons),
        FlyGymDecoder(state.n_neurons),
        decay=0.55,
        gain=0.45,
        sensory_indices=SENSORY_INDICES,
    )


def resolve_body_env(env: Any | None, *, try_real_flygym: bool) -> Any:
    if env is not None:
        return env
    if try_real_flygym:
        try:
            from fly_harness.flygym.wrapper import try_make_neuromechfly_sim

            return try_make_neuromechfly_sim()
        except Exception:
            return StubFlyGymEnv()
    return StubFlyGymEnv()


def build_composed_loop(
    env: Any | None = None,
    *,
    try_real_flygym: bool = False,
) -> ComposedLoop:
    """Wire FlyHarness.step to a FlyGym-shaped body. Default env needs no MuJoCo."""
    harness = build_composed_harness()
    body_env = resolve_body_env(env, try_real_flygym=try_real_flygym)
    body = FlyGymHarnessEnv(harness=harness, env=body_env)
    session = BodyAttachedSession(harness, body)
    body.reset()
    return ComposedLoop(harness=harness, body=body, session=session)


def run_scripted_loop(loop: ComposedLoop, steps: int = 3) -> list[dict[str, Any]]:
    """Drive the loop the way an MCP host would: repeated ``harness_step``-shaped calls."""
    loop.session.reset()
    schedule = (
        {"touch_left": 0.0, "touch_right": 0.0},
        {"touch_left": 1.0, "touch_right": 0.0},
        {"touch_left": 1.0, "touch_right": 0.0},
    )
    traces: list[dict[str, Any]] = []
    for i in range(max(1, steps)):
        traces.append(loop.session.step(**schedule[i % len(schedule)]))
    return traces


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=_BANNER)
    parser.add_argument(
        "--serve",
        action="store_true",
        help="run the optional FastMCP stdio server on this composed session",
    )
    parser.add_argument(
        "--try-flygym",
        action="store_true",
        help="try a real NeuroMechFly sim; fall back to the in-process stub",
    )
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)

    print(_BANNER)
    loop = build_composed_loop(try_real_flygym=args.try_flygym)
    print(f"n_neurons={loop.n_neurons} (fixture; not MaleCNS 166k)")
    body_kind = type(loop.body.env).__name__
    print(f"body={body_kind}")

    if args.serve:
        from fly_harness.mcp.server import create_mcp_server

        print("stdio MCP: tools harness_step / harness_reset / harness_status")
        create_mcp_server(loop.session).run()
        return

    traces = run_scripted_loop(loop, steps=args.steps)
    for i, row in enumerate(traces):
        joints = row["body_action"]["joints"]
        action = row["action"]
        print(
            f"step {i}: n_neurons={row['n_neurons']} "
            f"timestamp={row['timestamp']} "
            f"body_joints[:3]={np.asarray(joints)[:3]} "
            f"action_keys={sorted(action)}"
        )
    print("Pass --serve to attach the optional MCP extra to this same loop.")


if __name__ == "__main__":
    main()
