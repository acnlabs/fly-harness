"""In-process FlyBrain stand-in so tests and the example run without ~/fly-data."""

from __future__ import annotations

import numpy as np

FAKE_N_NEURONS = 32


class FakeFlyBrain:
    """Duck-typed ``FlyBrain``: ``step(inject=)``, ``reset``, ``n``, ``v``, ``dt``.

    Not MaleCNS. Default size is 32, not 166,700.
    """

    dt = 0.020

    def __init__(self, n_neurons: int = FAKE_N_NEURONS, *, seed: int = 0) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        self.n = int(n_neurons)
        self.batch = 1
        self.device = "cpu"
        self.cell_type = np.array(
            ["sens"] * 4 + ["inter"] * (self.n - 8) + ["descending_neuron"] * 4
        )
        self.side = np.array(["L"] * (self.n // 2) + ["R"] * (self.n - self.n // 2))
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.v = np.zeros((self.n, 1), dtype=np.float32)
        self.fired = np.empty(0, dtype=np.int64)
        self.steps = 0
        self._seed = 0 if seed is None else int(seed)

    def cells(self, types: list[str], side: str | None = None) -> np.ndarray:
        mask = np.isin(self.cell_type, types)
        if side:
            mask &= self.side == side
        return np.flatnonzero(mask)

    def stimulate(self, idx: np.ndarray, amount: float) -> None:
        self.v[np.asarray(idx)] += np.float32(amount)

    def step(self, eye_drive: np.ndarray | None = None, inject=()):
        for idx, amount in inject:
            self.stimulate(idx, amount)
        if eye_drive is not None:
            drive = np.asarray(eye_drive, dtype=np.float32).reshape(-1)
            n = min(drive.size, self.n)
            self.v[:n, 0] += drive[:n]
        fired = np.flatnonzero(self.v[:, 0] >= 1.0)
        self.v[fired, 0] = 0.0
        self.v *= np.float32(0.82)
        self.fired = fired
        self.steps += 1
        return fired
