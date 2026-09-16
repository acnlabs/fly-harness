"""``ModelBackend`` adapter around a running third-party ``FlyBrain``.

Does not rewrite flybrain. Does not download MaleCNS unless the caller
constructs ``FlyBrain()`` themselves (that package may download into
``~/fly-data``). ``from_installed(download=False)`` refuses if files are missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fly_harness.flybrain.detect import require_flybrain

DEFAULT_MODEL_ID = "flybrain.malecns-v1"
MALECNS_N_NEURONS = 166_700


def _inject_pairs(input_current: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Group nonzero currents into FlyBrain ``inject`` (idx, amount) pairs."""
    current = np.asarray(input_current, dtype=np.float64).reshape(-1)
    nz = np.flatnonzero(current)
    if nz.size == 0:
        return []
    pairs: list[tuple[np.ndarray, float]] = []
    values = current[nz]
    for amount in np.unique(values):
        idx = nz[values == amount]
        pairs.append((idx, float(amount)))
    return pairs


def _as_numpy(array: Any) -> np.ndarray:
    getter = getattr(array, "get", None)
    if callable(getter):
        array = getter()
    return np.asarray(array)


class FlyBrainBackend:
    """Dock a running ``FlyBrain`` (or duck-typed fake) as a ``ModelBackend``.

    ``tick`` turns a length-``n_neurons`` encoded current into
    ``FlyBrain.step(inject=...)`` and returns membrane voltages. This is a
    third-party LIF over a MaleCNS **dump**, not a Google-hosted sim, not the
    FlyWire website, and not ~160k "parameters" — MaleCNS v1.0 is **166,700
    neurons**.
    """

    DEFAULT_MODEL_ID = DEFAULT_MODEL_ID

    def __init__(
        self,
        brain: Any,
        *,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        if brain is None:
            raise ValueError("FlyBrainBackend needs a running brain instance")
        if not model_id:
            raise ValueError("model_id must be a non-empty string")
        n = getattr(brain, "n", None)
        if n is None:
            raise TypeError("brain must expose .n (neuron count)")
        self.brain = brain
        self._model_id = model_id
        self._n_neurons = int(n)
        self._timestamp = 0.0
        self.last_fired: np.ndarray = np.empty(0, dtype=np.int64)

    @classmethod
    def from_installed(
        cls,
        *,
        data: Any | None = None,
        download: bool = False,
        model_id: str = DEFAULT_MODEL_ID,
        **kwargs: Any,
    ) -> FlyBrainBackend:
        """Build from the third-party package. Never downloads unless ``download=True``."""
        require_flybrain()
        from flybrain import FlyBrain
        from flybrain.data import DATA, has_data

        path = DATA if data is None else data
        if not download and not has_data(path):
            raise FileNotFoundError(
                "flybrain MaleCNS files are not on disk "
                f"({path}). This extra will not download them. "
                "Run `flybrain download` yourself, then retry."
            )
        brain = FlyBrain(data=path, **kwargs)
        return cls(brain, model_id=model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def potentials(self) -> np.ndarray:
        return self._voltages()

    @property
    def dt(self) -> float:
        return float(getattr(self.brain, "dt", 1.0))

    def reset(self, potentials: np.ndarray | None = None) -> None:
        reset = getattr(self.brain, "reset", None)
        if callable(reset):
            reset()
        if potentials is not None:
            arr = np.asarray(potentials, dtype=np.float32).reshape(-1)
            if arr.shape != (self._n_neurons,):
                raise ValueError("potentials shape mismatch on reset")
            v = getattr(self.brain, "v", None)
            if v is None:
                raise TypeError("brain has no .v to restore potentials into")
            v[:, 0] = arr
        self._timestamp = 0.0
        self.last_fired = np.empty(0, dtype=np.int64)

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        current = np.asarray(input_current, dtype=np.float64)
        if current.shape != (self._n_neurons,):
            raise ValueError(
                f"encoded input length must be {self._n_neurons}, got {current.shape}"
            )
        fired = self.brain.step(inject=_inject_pairs(current))
        self.last_fired = np.asarray(_as_numpy(fired), dtype=np.int64).reshape(-1)
        self._timestamp += self.dt
        return self._voltages()

    def _voltages(self) -> np.ndarray:
        v = getattr(self.brain, "v", None)
        if v is None:
            out = np.zeros(self._n_neurons, dtype=np.float64)
            if self.last_fired.size:
                out[self.last_fired] = 1.0
            return out
        arr = _as_numpy(v)
        if arr.ndim == 2:
            arr = arr[:, 0]
        return np.asarray(arr, dtype=np.float64).reshape(-1)
