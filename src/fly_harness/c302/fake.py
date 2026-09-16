"""In-process C. elegans stand-in so tests list ``c302.celegans`` without OpenWorm.

Not OpenWorm Docker, not a hosted API, not a connectome download, and not a
consciousness claim. 302 is the hermaphrodite somatic neuron count.
"""

from __future__ import annotations

import numpy as np

C302_MODEL_ID = "c302.celegans"
C302_N_NEURONS = 302


class FakeC302:
    """Duck-typed deployed worm Model: ``tick`` / ``reset`` / ``n_neurons``.

    Implements the ``ModelBackend`` surface so biorouter can list
    ``c302.celegans`` in default CI. Not a fly-harness extra and not a
    third-party whole-brain package.
    """

    DEFAULT_MODEL_ID = C302_MODEL_ID
    N_NEURONS = C302_N_NEURONS

    def __init__(
        self,
        n_neurons: int = C302_N_NEURONS,
        *,
        model_id: str = C302_MODEL_ID,
        scale: float = 1.0,
        leak: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        if not model_id:
            raise ValueError("model_id must be a non-empty string")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._model_id = model_id
        self._n_neurons = int(n_neurons)
        self.scale = float(scale)
        self.leak = float(leak)
        self.dt = float(dt)
        self.potentials = np.zeros(self._n_neurons, dtype=np.float64)
        self._timestamp = 0.0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def reset(self, potentials: np.ndarray | None = None) -> None:
        if potentials is None:
            self.potentials.fill(0.0)
        else:
            arr = np.asarray(potentials, dtype=np.float64)
            if arr.shape != (self._n_neurons,):
                raise ValueError("potentials shape mismatch on reset")
            self.potentials[:] = arr
        self._timestamp = 0.0

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        current = np.asarray(input_current, dtype=np.float64)
        if current.shape != (self._n_neurons,):
            raise ValueError(
                f"encoded input length must be {self._n_neurons}, got {current.shape}"
            )
        self.potentials = self.leak * self.potentials + self.scale * current
        self._timestamp += self.dt
        return self.potentials.copy()
