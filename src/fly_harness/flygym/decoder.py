"""Decode harness potentials into FlyGym joint / adhesion / muscle commands."""

from __future__ import annotations

import numpy as np

from fly_harness.flygym.schema import (
    DOFS_PER_LEG,
    N_LEG_JOINTS,
    N_LEGS,
    FlyGymAction,
    _turn_joint_bias,
)
from fly_harness.protocols import BaseDecoder


class FlyGymDecoder(BaseDecoder):
    """Read motor-like potentials and emit a FlyGym action dict.

    ``rest_pose`` is the caller's neutral joint vector (FlyGym ``all_leg_dofs``
    order, length 42 by default). Potentials only add a small descending bias;
    they do not replace FlyGym's CPG or physics.
    """

    def __init__(
        self,
        n_neurons: int,
        rest_pose: np.ndarray | None = None,
        *,
        n_joints: int = N_LEG_JOINTS,
        include_adhesion: bool = True,
        include_muscle: bool = False,
        turn_gain: float = 0.12,
        motor_start: int | None = None,
        threshold: float = 0.05,
    ) -> None:
        super().__init__(n_neurons)
        self.n_joints = int(n_joints)
        if self.n_joints <= 0:
            raise ValueError("n_joints must be positive")
        if rest_pose is None:
            self.rest_pose = np.zeros(self.n_joints, dtype=np.float64)
        else:
            pose = np.asarray(rest_pose, dtype=np.float64).reshape(-1)
            if pose.shape != (self.n_joints,):
                raise ValueError(
                    f"rest_pose length {pose.shape[0]} must match n_joints={self.n_joints}"
                )
            self.rest_pose = pose
        self.include_adhesion = include_adhesion
        self.include_muscle = include_muscle
        self.turn_gain = float(turn_gain)
        self.threshold = float(threshold)
        if motor_start is None:
            self.motor_start = max(0, n_neurons - 8)
        else:
            self.motor_start = int(motor_start)

    def decode(self, potentials: np.ndarray) -> FlyGymAction:
        v = self._validate_potentials(potentials)
        motor = np.clip(v[self.motor_start :], 0.0, None)
        turn, forward, brake = self._descending(motor)

        joints = self.rest_pose + _turn_joint_bias(
            self.n_joints, turn, gain=self.turn_gain, dofs_per_leg=DOFS_PER_LEG
        )
        if brake > 0.5:
            joints = self.rest_pose.copy()

        adhesion = None
        if self.include_adhesion:
            grip = max(0.0, min(1.0, forward))
            if brake > 0.5:
                grip = 0.0
            adhesion = np.full(N_LEGS, grip, dtype=np.float64)

        muscle = None
        if self.include_muscle:
            if motor.size == 0:
                muscle = np.zeros(self.n_joints, dtype=np.float64)
            else:
                muscle = np.clip(np.resize(motor, self.n_joints), 0.0, 1.0)

        return FlyGymAction(joints=joints, adhesion=adhesion, muscle=muscle)

    def _descending(self, motor: np.ndarray) -> tuple[float, float, float]:
        if motor.size == 0:
            return 0.0, 0.0, 0.0
        if motor.size >= 8:
            turn_left = float(np.mean(motor[0:2]))
            forward = float(np.mean(motor[2:4]))
            turn_right = float(np.mean(motor[4:6]))
            brake = float(np.mean(motor[6:8]))
        elif motor.size >= 3:
            turn_left, forward, turn_right = (float(x) for x in motor[:3])
            brake = float(motor[-1]) if motor.size > 3 else 0.0
        else:
            return 0.0, float(np.mean(motor)), 0.0

        if turn_left > turn_right and turn_left > self.threshold:
            turn = -1.0
        elif turn_right > turn_left and turn_right > self.threshold:
            turn = 1.0
        else:
            turn = 0.0
        return turn, max(0.0, min(1.0, forward)), max(0.0, min(1.0, brake))
