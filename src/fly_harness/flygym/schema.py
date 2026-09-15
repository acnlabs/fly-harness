"""FlyGym / NeuroMechFly observation and action shapes — no third-party import.

These match the Gymnasium NeuroMechFly (FlyGym 1.x) dicts:

- observation ``joints`` is ``(3, n_dofs)``: angles, velocities, torques
- observation ``contact_forces`` is ``(n_sensors, 3)``
- action ``joints`` is ``(n_dofs,)`` position/muscle targets
- action ``adhesion`` is ``(6,)`` per-leg on/off in ``[0, 1]``

FlyGym 2.x ``Simulation`` uses the same numeric roles via
``get_joint_angles`` / ``set_actuator_inputs`` / ``set_leg_adhesion_states``.
This module does not reimplement FlyGym physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

N_LEGS = 6
DOFS_PER_LEG = 7  # Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1
N_LEG_JOINTS = N_LEGS * DOFS_PER_LEG  # 42 = FlyGym all_leg_dofs
LEG_ORDER: tuple[str, ...] = ("LF", "LM", "LH", "RF", "RM", "RH")
COXA_YAW_DOF = 2
INSTALL_HINT = "pip install 'fly-harness[flygym]'"


@dataclass(frozen=True)
class FlyGymAction:
    """Joint / muscle command ready for a FlyGym env or Simulation."""

    joints: np.ndarray
    adhesion: np.ndarray | None = None
    muscle: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", np.asarray(self.joints, dtype=np.float64).reshape(-1))
        if self.adhesion is not None:
            object.__setattr__(
                self, "adhesion", np.asarray(self.adhesion, dtype=np.float64).reshape(-1)
            )
            if self.adhesion.shape != (N_LEGS,):
                raise ValueError(f"adhesion must have shape ({N_LEGS},), got {self.adhesion.shape}")
        if self.muscle is not None:
            object.__setattr__(
                self, "muscle", np.asarray(self.muscle, dtype=np.float64).reshape(-1)
            )

    def as_env_dict(self) -> dict[str, np.ndarray]:
        """Gymnasium FlyGym ``env.step`` payload."""
        payload: dict[str, np.ndarray] = {"joints": self.joints.copy()}
        if self.adhesion is not None:
            payload["adhesion"] = self.adhesion.copy()
        if self.muscle is not None:
            payload["muscle"] = self.muscle.copy()
        return payload


def extract_joint_angles(observation: Any) -> np.ndarray:
    """Return 1-D joint angles from a FlyGym-style observation."""
    if isinstance(observation, np.ndarray):
        return np.asarray(observation, dtype=np.float64).reshape(-1)
    if hasattr(observation, "joint_angles") and not isinstance(observation, Mapping):
        return np.asarray(observation.joint_angles, dtype=np.float64).reshape(-1)
    if not isinstance(observation, Mapping):
        raise TypeError(
            "FlyGym observation must be a mapping, ndarray, or object with joint_angles"
        )

    if "joint_angles" in observation:
        return np.asarray(observation["joint_angles"], dtype=np.float64).reshape(-1)

    joints = observation.get("joints")
    if joints is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(joints, dtype=np.float64)
    if arr.ndim == 2:
        # FlyGym 1.x: row 0 = position (radians)
        return arr[0].reshape(-1)
    return arr.reshape(-1)


def extract_contact_forces(observation: Any) -> np.ndarray:
    """Return contact-force array ``(n_sensors, 3)`` or empty."""
    if isinstance(observation, Mapping) and "contact_forces" in observation:
        arr = np.asarray(observation["contact_forces"], dtype=np.float64)
        if arr.ndim == 1:
            if arr.size % 3 == 0:
                return arr.reshape(-1, 3)
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            return arr
        return arr.reshape(-1, arr.shape[-1])
    return np.zeros((0, 3), dtype=np.float64)


def contact_magnitudes_by_leg(
    contact_forces: np.ndarray, n_legs: int = N_LEGS
) -> np.ndarray:
    """Mean contact magnitude per leg (LF, LM, LH, RF, RM, RH)."""
    forces = np.asarray(contact_forces, dtype=np.float64)
    if forces.size == 0:
        return np.zeros(n_legs, dtype=np.float64)
    if forces.ndim == 1:
        mags = np.abs(forces)
    else:
        mags = np.linalg.norm(forces, axis=-1)
    n_sensors = int(mags.shape[0])
    if n_sensors == n_legs:
        return mags.astype(np.float64, copy=False)
    if n_sensors % n_legs == 0:
        grouped = mags.reshape(n_legs, -1)
        return grouped.mean(axis=1)
    # Truncate or pad to n_legs
    out = np.zeros(n_legs, dtype=np.float64)
    n = min(n_legs, n_sensors)
    out[:n] = mags[:n]
    return out


def left_right_contact(contact_forces: np.ndarray) -> tuple[float, float]:
    """Average contact on left legs (LF/LM/LH) vs right (RF/RM/RH)."""
    by_leg = contact_magnitudes_by_leg(contact_forces)
    left = float(np.mean(by_leg[:3]))
    right = float(np.mean(by_leg[3:]))
    return left, right


def coerce_action(action: Any, rest_pose: np.ndarray | None = None) -> FlyGymAction:
    """Accept ``FlyGymAction``, env dicts, or fixture ``ReflexAction``-like objects."""
    if isinstance(action, FlyGymAction):
        return action
    if isinstance(action, Mapping):
        joints = action.get("joints", action.get("muscle"))
        if joints is None:
            raise TypeError("action dict must include 'joints' or 'muscle'")
        return FlyGymAction(
            joints=np.asarray(joints, dtype=np.float64),
            adhesion=action.get("adhesion"),
            muscle=action.get("muscle"),
        )
    if rest_pose is not None and hasattr(action, "turn") and hasattr(action, "forward"):
        return reflex_like_to_flygym(action, rest_pose)
    raise TypeError(
        "expected FlyGymAction, dict with 'joints', or a reflex-like action "
        f"with turn/forward; got {type(action)!r}"
    )


def reflex_like_to_flygym(action: Any, rest_pose: np.ndarray) -> FlyGymAction:
    """Map fixture turn/forward/brake onto a rest pose + adhesion."""
    rest = np.asarray(rest_pose, dtype=np.float64).reshape(-1)
    turn = float(getattr(action, "turn", 0.0))
    forward = float(getattr(action, "forward", 0.0))
    brake = float(getattr(action, "brake", 0.0))
    joints = rest + _turn_joint_bias(rest.shape[0], turn)
    if brake > 0.5:
        joints = rest.copy()
    adhesion = np.full(N_LEGS, max(0.0, min(1.0, forward)), dtype=np.float64)
    if brake > 0.5:
        adhesion[:] = 0.0
    return FlyGymAction(joints=joints, adhesion=adhesion)


def _turn_joint_bias(
    n_joints: int, turn: float, *, gain: float = 0.12, dofs_per_leg: int = DOFS_PER_LEG
) -> np.ndarray:
    """Antisymmetric coxa-yaw bias: +turn yaws left legs out, right legs the other way."""
    bias = np.zeros(n_joints, dtype=np.float64)
    if n_joints < 2:
        return bias
    if n_joints % N_LEGS == 0:
        stride = n_joints // N_LEGS
        yaw = min(COXA_YAW_DOF, stride - 1)
        for leg in range(3):
            bias[leg * stride + yaw] = turn * gain
        for leg in range(3, 6):
            bias[leg * stride + yaw] = -turn * gain
        return bias
    mid = n_joints // 2
    bias[:mid] = turn * gain
    bias[mid:] = -turn * gain
    return bias
