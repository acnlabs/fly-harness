"""Encoder/decoder implementations for the reflex demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fly_harness.demo.connectome import (
    N_NEURONS,
    _MOTOR_BRAKE,
    _MOTOR_FORWARD,
    _MOTOR_TURN_LEFT,
    _MOTOR_TURN_RIGHT,
    _SENS_LEFT,
    _SENS_RIGHT,
)
from fly_harness.protocols import BaseDecoder, BaseEncoder


@dataclass(frozen=True)
class TouchObservation:
    """Simple touch sensors on left and right."""

    touch_left: float = 0.0
    touch_right: float = 0.0


@dataclass(frozen=True)
class ReflexAction:
    """Discrete motor command from the demo decoder."""

    turn: int  # -1 left, 0 straight, +1 right
    forward: float  # 0..1 throttle
    brake: float  # 0..1


class ReflexEncoder(BaseEncoder):
    """Maps touch observations to currents on sensory neuron indices."""

    def __init__(self) -> None:
        super().__init__(N_NEURONS)

    def encode(self, observation: Any) -> np.ndarray:
        currents = self._zeros()
        if isinstance(observation, TouchObservation):
            left = float(observation.touch_left)
            right = float(observation.touch_right)
        elif isinstance(observation, dict):
            left = float(observation.get("touch_left", 0.0))
            right = float(observation.get("touch_right", 0.0))
        else:
            raise TypeError(
                "ReflexEncoder expects TouchObservation or dict with touch_left/touch_right"
            )

        for idx in _SENS_LEFT:
            currents[idx] = left
        for idx in _SENS_RIGHT:
            currents[idx] = right
        return currents


class ReflexDecoder(BaseDecoder):
    """Reads motor neuron pools and picks a reflex action."""

    def __init__(self, threshold: float = 0.15) -> None:
        super().__init__(N_NEURONS)
        self.threshold = threshold

    def decode(self, potentials: np.ndarray) -> ReflexAction:
        v = self._validate_potentials(potentials)

        turn_left = float(np.mean(np.clip(v[list(_MOTOR_TURN_LEFT)], 0.0, None)))
        turn_right = float(np.mean(np.clip(v[list(_MOTOR_TURN_RIGHT)], 0.0, None)))
        forward = float(np.mean(np.clip(v[list(_MOTOR_FORWARD)], 0.0, None)))
        brake = float(np.mean(np.clip(v[list(_MOTOR_BRAKE)], 0.0, None)))

        if turn_left > turn_right and turn_left > self.threshold:
            turn = -1
        elif turn_right > turn_left and turn_right > self.threshold:
            turn = 1
        else:
            turn = 0

        forward = max(0.0, min(1.0, forward))
        brake = max(0.0, min(1.0, brake))
        if brake > 0.5:
            forward *= 0.25

        return ReflexAction(turn=turn, forward=forward, brake=brake)
