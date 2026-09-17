"""Swappable Model backends the harness docks to.

Two docking modes (a port, not a marketplace):

1. **Direct** — ``DirectBioSimBackend``: one backend per vendor/deployed sim.
2. **Router** — ``BioSimRouter``: one entry that selects a backend by
   ``model_id`` (OpenRouter-shaped ``model`` field). Unknown ids error.

The default in-process rate / leaky-integrator circuit is
``InProcessLifBackend``. There is no hosted bio-sim OpenRouter here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fly_harness.brain_state import BrainState
from fly_harness.protocols import ModelBackend

DEFAULT_LIF_MODEL_ID = "fly-harness.in-process-lif"


class UnknownModelError(LookupError):
    """Raised when ``model_id`` is not registered and cannot be resolved."""

    def __init__(self, model_id: str, registered: list[str] | None = None) -> None:
        self.model_id = model_id
        self.registered = list(registered or [])
        names = ", ".join(repr(name) for name in self.registered) if self.registered else "(none)"
        super().__init__(
            f"unknown model_id {model_id!r}; registered: {names}. "
            "BioSimRouter is an in-process port, not a marketplace."
        )


class SnapshotUnsupportedError(NotImplementedError):
    """Raised when ``snapshot`` / ``restore`` has no mapping on this backend.

    Optional ``ModelBackend`` port, same family as ``reset``. Missing is an
    explicit error, not a silent no-op. The harness does not invent vendor
    checkpoint formats (flybrain / c302 / NEURON / remote HTTP).
    """

    def __init__(
        self,
        backend: Any | None = None,
        *,
        operation: str = "snapshot",
        detail: str | None = None,
    ) -> None:
        name = type(backend).__name__ if backend is not None else "ModelBackend"
        model_id = getattr(backend, "model_id", None)
        id_part = f" ({model_id!r})" if model_id else ""
        message = detail or (
            f"{name}{id_part} has no {operation}() mapping. "
            "snapshot/restore is an optional ModelBackend port; "
            "missing is an explicit error, not a silent no-op. "
            "The harness does not invent vendor checkpoint formats."
        )
        self.operation = operation
        self.backend_type = name
        super().__init__(message)


def invoke_snapshot(backend: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``backend.snapshot`` or raise ``SnapshotUnsupportedError``."""
    return _invoke_checkpoint(backend, "snapshot", *args, **kwargs)


def invoke_restore(backend: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``backend.restore`` or raise ``SnapshotUnsupportedError``."""
    return _invoke_checkpoint(backend, "restore", *args, **kwargs)


def _invoke_checkpoint(backend: Any, operation: str, *args: Any, **kwargs: Any) -> Any:
    fn = getattr(backend, operation, None)
    if not callable(fn):
        raise SnapshotUnsupportedError(backend, operation=operation)
    try:
        return fn(*args, **kwargs)
    except SnapshotUnsupportedError:
        raise
    except NotImplementedError as exc:
        raise SnapshotUnsupportedError(
            backend,
            operation=operation,
            detail=str(exc) or None,
        ) from exc


def _as_current(input_current: np.ndarray, n_neurons: int) -> np.ndarray:
    current = np.asarray(input_current, dtype=np.float64)
    if current.shape != (n_neurons,):
        raise ValueError(
            f"encoded input length must be {n_neurons}, got {current.shape}"
        )
    return current


class InProcessLifBackend:
    """Default Model: the existing in-process sparse rate / leaky-integrator.

    This is the v0.2 dynamics extracted from ``FlyHarness.step``. It is a
    small leaky rate circuit, not a biophysical Hodgkin–Huxley cell model.
    """

    DEFAULT_MODEL_ID = DEFAULT_LIF_MODEL_ID

    def __init__(
        self,
        state: BrainState,
        *,
        decay: float = 0.2,
        gain: float = 0.35,
        dt: float = 1.0,
        sensory_indices: tuple[int, ...] | None = None,
        model_id: str = DEFAULT_LIF_MODEL_ID,
    ) -> None:
        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be in [0, 1]")
        if gain <= 0.0:
            raise ValueError("gain must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if not model_id:
            raise ValueError("model_id must be a non-empty string")

        self.state = state
        self.decay = decay
        self.gain = gain
        self.dt = dt
        self.sensory_indices = sensory_indices
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def n_neurons(self) -> int:
        return self.state.n_neurons

    @property
    def timestamp(self) -> float:
        return self.state.timestamp

    @property
    def potentials(self) -> np.ndarray:
        return self.state.potentials

    def reset(self, potentials: np.ndarray | None = None) -> None:
        if potentials is None:
            self.state.potentials.fill(0.0)
        else:
            arr = np.asarray(potentials, dtype=np.float64)
            if arr.shape != self.state.potentials.shape:
                raise ValueError("potentials shape mismatch on reset")
            self.state.potentials[:] = arr
        self.state.timestamp = 0.0

    def snapshot(self, path: str | Path | None = None) -> dict[str, Any]:
        """Checkpoint this toy LIF via ``BrainState`` JSON.

        Fixture format (``BrainState.save`` / ``load``), not a universal Model
        file. ``path`` writes the JSON; the dict is always returned.
        """
        payload = self.state.to_dict()
        if path is not None:
            self.state.save(path)
        return payload

    def restore(self, snapshot: str | Path | dict[str, Any]) -> None:
        """Load a toy-LIF ``BrainState`` checkpoint into this backend in place."""
        if isinstance(snapshot, dict):
            loaded = BrainState.from_dict(snapshot)
        else:
            loaded = BrainState.load(snapshot)
        if loaded.n_neurons != self.n_neurons:
            raise ValueError(
                f"snapshot n_neurons {loaded.n_neurons} does not match "
                f"backend n_neurons {self.n_neurons}"
            )
        self.state.potentials = np.asarray(loaded.potentials, dtype=np.float64).copy()
        self.state.weights = loaded.weights.copy()
        self.state.timestamp = float(loaded.timestamp)
        self.state.neuron_labels = loaded.neuron_labels

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        current = _as_current(input_current, self.n_neurons)

        if self.sensory_indices is not None:
            for idx in self.sensory_indices:
                self.state.potentials[idx] = current[idx]

        # weights[pre, post]: postsynaptic drive uses transpose
        drive = self.state.weights.T @ self._activation(self.state.potentials)
        self.state.potentials = (1.0 - self.decay) * self.state.potentials + self.gain * drive
        if self.sensory_indices is None:
            self.state.potentials += current
        else:
            for idx in self.sensory_indices:
                self.state.potentials[idx] = current[idx]
        self.state.timestamp += self.dt
        return self.state.potentials.copy()

    @staticmethod
    def _activation(potentials: np.ndarray) -> np.ndarray:
        """Simple rectified rate nonlinearity."""
        return np.clip(potentials, 0.0, None)


class FakeDeployedSim:
    """In-process stand-in for a vendor-deployed bio-sim.

    Not a connectome dump and not a 140k dense matrix. Tests and local
    docking use this instead of FlyWire / CAVE / neuPrint clients.
    """

    def __init__(
        self,
        n_neurons: int,
        *,
        model_id: str = "fake.deployed",
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

    def snapshot(self, *args: Any, **kwargs: Any) -> Any:
        raise SnapshotUnsupportedError(self, operation="snapshot")

    def restore(self, *args: Any, **kwargs: Any) -> Any:
        raise SnapshotUnsupportedError(self, operation="restore")

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        current = _as_current(input_current, self._n_neurons)
        self.potentials = self.leak * self.potentials + self.scale * current
        self._timestamp += self.dt
        return self.potentials.copy()


class DirectBioSimBackend:
    """Direct docking: one backend per vendor / deployed sim.

    Pass an in-process sim (``FakeDeployedSim`` is enough for tests) or a
    ``url`` for a thin HTTP client. No FlyWire / CAVE / neuPrint clients.
    """

    def __init__(
        self,
        sim: Any | None = None,
        *,
        url: str | None = None,
        model_id: str | None = None,
        n_neurons: int | None = None,
        timeout: float = 5.0,
        scale: float = 1.0,
        leak: float = 0.0,
    ) -> None:
        if sim is not None and url is not None:
            raise ValueError("pass an in-process sim or a url, not both")
        self._model_id_override = model_id
        if url is not None:
            from fly_harness.http_backend import HttpModelBackend

            if not model_id:
                raise ValueError("url docking requires model_id")
            self._impl: Any = HttpModelBackend(
                url,
                model_id=model_id,
                n_neurons=n_neurons,
                timeout=timeout,
            )
        elif sim is None:
            if n_neurons is None:
                raise ValueError(
                    "DirectBioSimBackend needs a sim, url, or n_neurons "
                    "to build FakeDeployedSim"
                )
            self._impl = FakeDeployedSim(
                n_neurons,
                model_id=model_id or "fake.deployed",
                scale=scale,
                leak=leak,
            )
            self._model_id_override = None
        else:
            self._impl = sim

    @property
    def model_id(self) -> str:
        if self._model_id_override:
            return self._model_id_override
        return str(self._impl.model_id)

    @property
    def n_neurons(self) -> int:
        return int(self._impl.n_neurons)

    @property
    def timestamp(self) -> float:
        return float(self._impl.timestamp)

    @property
    def potentials(self) -> np.ndarray:
        pots = getattr(self._impl, "potentials", None)
        if pots is None:
            return np.zeros(self.n_neurons, dtype=np.float64)
        return np.asarray(pots, dtype=np.float64)

    @property
    def state(self) -> Any | None:
        return getattr(self._impl, "state", None)

    def reset(self, potentials: np.ndarray | None = None) -> None:
        self._impl.reset(potentials)

    def snapshot(self, *args: Any, **kwargs: Any) -> Any:
        """Forward only when the wrapped sim already has a snapshot mapping."""
        return invoke_snapshot(self._impl, *args, **kwargs)

    def restore(self, *args: Any, **kwargs: Any) -> Any:
        """Forward only when the wrapped sim already has a restore mapping."""
        return invoke_restore(self._impl, *args, **kwargs)

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        return np.asarray(self._impl.tick(input_current), dtype=np.float64)


class BioSimRouter:
    """Router docking: one entry that selects a ``ModelBackend`` by model id.

    OpenRouter-shaped: callers send a ``model`` / ``model_id``. This object
    is an in-process map, optionally forwarding to a remote router URL with
    a ``model`` field. It is not a hosted marketplace, billing system, or
    multi-tenant gateway.
    """

    def __init__(
        self,
        backends: Mapping[str, ModelBackend] | None = None,
        *,
        model_id: str | None = None,
        remote_url: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._backends: dict[str, ModelBackend] = {}
        for key, backend in dict(backends or {}).items():
            self.register(key, backend)
        self._remote_url = remote_url.rstrip("/") if remote_url else None
        self._timeout = float(timeout)
        self._active: ModelBackend | None = None
        if model_id is not None:
            self.use(model_id)

    def register(self, model_id: str, backend: ModelBackend) -> None:
        if not model_id:
            raise ValueError("model_id must be a non-empty string")
        self._backends[model_id] = backend

    def registered_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def select(self, model_id: str) -> ModelBackend:
        """Resolve ``model_id`` to a backend. Unknown ids raise ``UnknownModelError``."""
        if not model_id:
            raise UnknownModelError(model_id, registered=sorted(self._backends))
        backend = self._backends.get(model_id)
        if backend is not None:
            return backend
        if self._remote_url is not None:
            from fly_harness.http_backend import HttpModelBackend

            return HttpModelBackend(
                self._remote_url,
                model_id=model_id,
                timeout=self._timeout,
            )
        raise UnknownModelError(model_id, registered=sorted(self._backends))

    def use(self, model_id: str) -> ModelBackend:
        """Select the active backend for subsequent ``tick`` / ``reset`` calls."""
        self._active = self.select(model_id)
        return self._active

    @property
    def active(self) -> ModelBackend:
        if self._active is None:
            raise RuntimeError(
                "BioSimRouter has no selected model_id; call use(model_id) "
                "or pass model_id=... when constructing"
            )
        return self._active

    @property
    def model_id(self) -> str:
        return self.active.model_id

    @property
    def n_neurons(self) -> int:
        return self.active.n_neurons

    @property
    def timestamp(self) -> float:
        return self.active.timestamp

    @property
    def potentials(self) -> np.ndarray:
        pots = getattr(self.active, "potentials", None)
        if pots is None:
            return np.zeros(self.n_neurons, dtype=np.float64)
        return np.asarray(pots, dtype=np.float64)

    def reset(self, potentials: np.ndarray | None = None) -> None:
        self.active.reset(potentials)

    def snapshot(self, *args: Any, **kwargs: Any) -> Any:
        """Forward only when the selected backend already has a snapshot mapping."""
        return invoke_snapshot(self.active, *args, **kwargs)

    def restore(self, *args: Any, **kwargs: Any) -> Any:
        """Forward only when the selected backend already has a restore mapping."""
        return invoke_restore(self.active, *args, **kwargs)

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        return np.asarray(self.active.tick(input_current), dtype=np.float64)
