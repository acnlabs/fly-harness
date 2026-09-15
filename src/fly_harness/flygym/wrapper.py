"""Thin env wrapper: FlyGym obs -> FlyHarness.step -> FlyGym joint/muscle commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fly_harness.flygym.schema import FlyGymAction, coerce_action
from fly_harness.harness import FlyHarness, StepResult


@dataclass
class BodyStepResult:
    """One coupled harness + body step."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    harness: StepResult
    action: FlyGymAction


class FlyGymHarnessEnv:
    """Drive a FlyGym / NeuroMechFly env with ``FlyHarness.step``.

    The env is constructed by the caller (do not rewrite FlyGym). Gymnasium-style
    ``reset`` / ``step(action_dict)`` is the default. FlyGym 2.x ``Simulation``
    objects are accepted when ``set_actuator_inputs`` is present.
    """

    def __init__(
        self,
        env: Any,
        harness: FlyHarness,
        *,
        rest_pose: np.ndarray | None = None,
        fly_name: str = "0",
        actuator_type: Any = None,
    ) -> None:
        self.env = env
        self.harness = harness
        self.fly_name = fly_name
        self.actuator_type = actuator_type
        decoder_pose = getattr(harness.decoder, "rest_pose", None)
        self.rest_pose = (
            np.asarray(rest_pose, dtype=np.float64).reshape(-1)
            if rest_pose is not None
            else decoder_pose
        )
        self._last_obs: Any = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        reset_fn = getattr(self.env, "reset", None)
        if callable(reset_fn):
            obs, info = _unpack_reset(reset_fn(**kwargs))
        else:
            obs, info = None, {}
        if obs is None and hasattr(self.env, "get_joint_angles"):
            obs = observation_from_simulation(self.env, self.fly_name)
        self._last_obs = obs
        return obs, info

    def step(self, observation: Any = None) -> BodyStepResult:
        obs_in = self._last_obs if observation is None else observation
        if obs_in is None:
            raise RuntimeError("call reset() before step(), or pass an observation")
        harness_result = self.harness.step(obs_in)
        action = coerce_action(harness_result.action, rest_pose=self.rest_pose)
        raw = apply_flygym_action(
            self.env,
            action,
            fly_name=self.fly_name,
            actuator_type=self.actuator_type,
        )
        obs, reward, terminated, truncated, info = _unpack_step(raw)
        if obs is None and hasattr(self.env, "get_joint_angles"):
            obs = observation_from_simulation(self.env, self.fly_name)
        self._last_obs = obs
        return BodyStepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            harness=harness_result,
            action=action,
        )


def apply_flygym_action(
    target: Any,
    action: FlyGymAction | dict[str, np.ndarray],
    *,
    fly_name: str = "0",
    actuator_type: Any = None,
) -> Any:
    """Send joint/adhesion/muscle commands to a FlyGym env or Simulation.

    Gymnasium NeuroMechFly: ``target.step({"joints", "adhesion"})``.
    FlyGym 2.x Simulation: ``set_actuator_inputs`` + optional adhesion/tendon, then physics ``step``.
    """
    payload = action.as_env_dict() if isinstance(action, FlyGymAction) else dict(action)

    if callable(getattr(target, "set_actuator_inputs", None)):
        if actuator_type is None:
            raise TypeError(
                "FlyGym 2.x Simulation.step needs actuator_type "
                "(pass the enum/object FlyGym expects; this adapter does not invent one)"
            )
        target.set_actuator_inputs(fly_name, actuator_type, payload["joints"])
        adhesion = payload.get("adhesion")
        if adhesion is not None and callable(getattr(target, "set_leg_adhesion_states", None)):
            target.set_leg_adhesion_states(fly_name, adhesion)
        muscle = payload.get("muscle")
        if muscle is not None and callable(getattr(target, "set_tendon_actuator_inputs", None)):
            target.set_tendon_actuator_inputs(fly_name, muscle)
        if callable(getattr(target, "step", None)):
            target.step()
        return None

    if not callable(getattr(target, "step", None)):
        raise TypeError(
            "body target must implement Gymnasium step(action) or "
            "FlyGym 2.x set_actuator_inputs(...)"
        )
    return target.step(payload)


def observation_from_simulation(sim: Any, fly_name: str) -> dict[str, np.ndarray]:
    """Build a dict observation from FlyGym 2.x Simulation getters."""
    angles = np.asarray(sim.get_joint_angles(fly_name), dtype=np.float64)
    obs: dict[str, np.ndarray] = {"joint_angles": angles, "joints": angles}
    getter = getattr(sim, "get_ground_contact_info", None)
    if callable(getter):
        packed = getter(fly_name)
        if isinstance(packed, tuple) and len(packed) >= 2:
            obs["contact_forces"] = np.asarray(packed[1], dtype=np.float64)
    return obs


def _unpack_reset(raw: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(raw, tuple) and len(raw) == 2:
        obs, info = raw
        return obs, dict(info) if info is not None else {}
    return raw, {}


def _unpack_step(raw: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    if raw is None:
        return None, 0.0, False, False, {}
    if isinstance(raw, tuple):
        if len(raw) == 5:
            obs, reward, terminated, truncated, info = raw
            return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})
        if len(raw) == 4:
            obs, reward, done, info = raw
            return obs, float(reward), bool(done), False, dict(info or {})
    return raw, 0.0, False, False, {}


def try_make_neuromechfly_sim() -> Any:
    """Best-effort Gymnasium NeuroMechFly sim. Raises ImportError if FlyGym is missing.

    Headless: empty camera list. This constructs FlyGym objects; it does not copy
    or reimplement the simulator.
    """
    import os

    from fly_harness.flygym.detect import import_flygym_module, require_flygym

    require_flygym()
    os.environ.setdefault("SKIP_RENDERING", "true")
    module = import_flygym_module()
    try:
        Fly = module.Fly
        SingleFlySimulation = module.SingleFlySimulation
        arena_mod = __import__(f"{module.__name__}.arena", fromlist=["FlatTerrain"])
        FlatTerrain = arena_mod.FlatTerrain
    except AttributeError as exc:
        raise ImportError(
            "found a FlyGym package but not the Gymnasium NeuroMechFly API "
            "(Fly, SingleFlySimulation). Install flygym-gymnasium via "
            "pip install 'fly-harness[flygym]'"
        ) from exc

    fly = Fly(init_pose="stretch", control="position", enable_adhesion=True)
    try:
        return SingleFlySimulation(fly=fly, cameras=[], arena=FlatTerrain())
    except TypeError:
        return SingleFlySimulation(fly=fly, cameras=[])
