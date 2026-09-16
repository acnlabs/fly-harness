"""Encoder/Decoder for docking FlyBrain through FlyHarness.step.

Maps a small inject/readout observation onto a length-``n_neurons`` current
vector. Does not load MaleCNS and does not rewrite flybrain.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fly_harness.protocols import BaseDecoder, BaseEncoder


class FlyBrainInjectEncoder(BaseEncoder):
    """Observation → current vector. Named channels map onto neuron indices."""

    def __init__(
        self,
        n_neurons: int,
        channels: Mapping[str, np.ndarray | tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__(n_neurons)
        self.channels = {
            str(name): np.asarray(idx, dtype=np.int64).reshape(-1)
            for name, idx in dict(channels or {}).items()
        }

    def encode(self, observation: Any) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            arr = np.asarray(observation, dtype=np.float64).reshape(-1)
            if arr.shape != (self.n_neurons,):
                raise ValueError(
                    f"expected currents shape ({self.n_neurons},), got {arr.shape}"
                )
            return arr
        if not isinstance(observation, Mapping):
            raise TypeError(
                "FlyBrainInjectEncoder expects a current vector or a dict of "
                "channel amounts / an 'inject' map"
            )
        currents = self._zeros()
        inject = observation.get("inject")
        if isinstance(inject, Mapping):
            for key, amount in inject.items():
                currents[int(key)] = float(amount)
        for name, idx in self.channels.items():
            if name in observation and name != "inject":
                currents[idx] = float(observation[name])
        return currents


class FlyBrainReadoutDecoder(BaseDecoder):
    """Mean voltage over readout indices (typically descending neurons)."""

    def __init__(
        self,
        n_neurons: int,
        readout_indices: np.ndarray | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(n_neurons)
        if readout_indices is None:
            self.readout_indices = np.arange(max(0, n_neurons - 4), n_neurons, dtype=np.int64)
        else:
            self.readout_indices = np.asarray(readout_indices, dtype=np.int64).reshape(-1)

    def decode(self, potentials: np.ndarray) -> dict[str, float | int]:
        v = self._validate_potentials(potentials)
        pool = v[self.readout_indices] if self.readout_indices.size else v
        mean = float(np.mean(pool))
        left = float(np.mean(pool[: max(1, pool.size // 2)]))
        right = float(np.mean(pool[pool.size // 2 :]))
        turn = 0
        if right - left > 1e-6:
            turn = 1
        elif left - right > 1e-6:
            turn = -1
        return {
            "readout_mean": mean,
            "turn": turn,
            "n_neurons": int(self.n_neurons),
        }
