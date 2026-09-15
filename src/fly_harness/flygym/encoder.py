"""Encode FlyGym / NeuroMechFly observations into harness input currents."""

from __future__ import annotations

from typing import Any

import numpy as np

from fly_harness.flygym.schema import (
    extract_contact_forces,
    extract_joint_angles,
    left_right_contact,
)
from fly_harness.protocols import BaseEncoder


class FlyGymEncoder(BaseEncoder):
    """Map joint angles and contact forces onto a length-``n_neurons`` current vector.

    Left vs right tarsal contact is written onto the first four neurons (the
    24-neuron reflex fixture's touch sensors) when ``n_neurons >= 4``. Remaining
    neurons receive subsampled proprioception. This does not load a whole-brain
    connectome; it only fills the harness the caller already constructed.
    """

    def __init__(
        self,
        n_neurons: int,
        *,
        contact_gain: float = 1.0,
        proprio_gain: float = 0.25,
        left_indices: tuple[int, ...] = (0, 1),
        right_indices: tuple[int, ...] = (2, 3),
    ) -> None:
        super().__init__(n_neurons)
        self.contact_gain = float(contact_gain)
        self.proprio_gain = float(proprio_gain)
        self.left_indices = left_indices
        self.right_indices = right_indices

    def encode(self, observation: Any) -> np.ndarray:
        currents = self._zeros()
        angles = extract_joint_angles(observation)
        contact = extract_contact_forces(observation)
        left, right = left_right_contact(contact)

        for idx in self.left_indices:
            if 0 <= idx < self.n_neurons:
                currents[idx] = self.contact_gain * left
        for idx in self.right_indices:
            if 0 <= idx < self.n_neurons:
                currents[idx] = self.contact_gain * right

        proprio = self.proprio_gain * np.asarray(angles, dtype=np.float64).reshape(-1)
        occupied = set(self.left_indices) | set(self.right_indices)
        free = [i for i in range(self.n_neurons) if i not in occupied]
        if free and proprio.size:
            tiled = np.resize(proprio, len(free))
            currents[np.array(free, dtype=int)] = tiled
        elif proprio.size and self.n_neurons:
            currents[:] = np.resize(proprio, self.n_neurons)
        return currents
