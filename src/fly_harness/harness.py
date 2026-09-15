"""FlyHarness microkernel: one observation-to-action step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fly_harness.brain_state import BrainState
from fly_harness.protocols import Decoder, Encoder


@dataclass
class StepResult:
    """Outcome of a single harness step."""

    action: Any
    potentials: np.ndarray
    timestamp: float


class FlyHarness:
    """Sparse rate-based microkernel stepping observations through a connectome."""

    def __init__(
        self,
        state: BrainState,
        encoder: Encoder,
        decoder: Decoder,
        *,
        decay: float = 0.2,
        gain: float = 0.35,
        dt: float = 1.0,
        sensory_indices: tuple[int, ...] | None = None,
    ) -> None:
        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be in [0, 1]")
        if gain <= 0.0:
            raise ValueError("gain must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.state = state
        self.encoder = encoder
        self.decoder = decoder
        self.decay = decay
        self.gain = gain
        self.dt = dt
        self.sensory_indices = sensory_indices

    def reset(self, potentials: np.ndarray | None = None) -> None:
        if potentials is None:
            self.state.potentials.fill(0.0)
        else:
            arr = np.asarray(potentials, dtype=np.float64)
            if arr.shape != self.state.potentials.shape:
                raise ValueError("potentials shape mismatch on reset")
            self.state.potentials[:] = arr
        self.state.timestamp = 0.0

    def step(self, observation: Any) -> StepResult:
        """Encode observation, advance dynamics once, decode action."""
        input_current = np.asarray(self.encoder.encode(observation), dtype=np.float64)
        if input_current.shape != self.state.potentials.shape:
            raise ValueError(
                "encoder output length must match BrainState.n_neurons "
                f"({self.state.n_neurons}), got {input_current.shape}"
            )

        if self.sensory_indices is not None:
            for idx in self.sensory_indices:
                self.state.potentials[idx] = input_current[idx]

        # weights[pre, post]: postsynaptic drive uses transpose
        drive = self.state.weights.T @ self._activation(self.state.potentials)
        self.state.potentials = (1.0 - self.decay) * self.state.potentials + self.gain * drive
        if self.sensory_indices is None:
            self.state.potentials += input_current
        else:
            for idx in self.sensory_indices:
                self.state.potentials[idx] = input_current[idx]
        self.state.timestamp += self.dt

        action = self.decoder.decode(self.state.potentials)
        return StepResult(
            action=action,
            potentials=self.state.potentials.copy(),
            timestamp=self.state.timestamp,
        )

    @staticmethod
    def _activation(potentials: np.ndarray) -> np.ndarray:
        """Simple rectified rate nonlinearity."""
        return np.clip(potentials, 0.0, None)
