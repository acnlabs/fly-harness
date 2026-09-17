"""FlyHarness microkernel: one observation-to-action step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fly_harness.backend import InProcessLifBackend, invoke_restore, invoke_snapshot
from fly_harness.brain_state import BrainState
from fly_harness.protocols import Decoder, Encoder, ModelBackend


@dataclass
class StepResult:
    """Outcome of a single harness step."""

    action: Any
    potentials: np.ndarray
    timestamp: float


class _BackendStateView:
    """Minimal ``.n_neurons`` / ``.timestamp`` surface for non-LIF backends."""

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    @property
    def n_neurons(self) -> int:
        return int(self._backend.n_neurons)

    @property
    def timestamp(self) -> float:
        return float(self._backend.timestamp)

    @property
    def potentials(self) -> np.ndarray:
        pots = getattr(self._backend, "potentials", None)
        if pots is None:
            return np.zeros(self.n_neurons, dtype=np.float64)
        return np.asarray(pots, dtype=np.float64)


class FlyHarness:
    """Sparse microkernel: encode -> ModelBackend.tick -> decode.

    Default construction ``FlyHarness(state, encoder, decoder)`` wraps the
    in-process LIF/rate circuit. Pass ``backend=`` (direct sim or router)
    to dock to another running Model. ``step(obs) -> action`` is unchanged.
    """

    def __init__(
        self,
        state: BrainState | ModelBackend | None = None,
        encoder: Encoder | None = None,
        decoder: Decoder | None = None,
        *,
        backend: ModelBackend | None = None,
        decay: float = 0.2,
        gain: float = 0.35,
        dt: float = 1.0,
        sensory_indices: tuple[int, ...] | None = None,
        model_id: str | None = None,
    ) -> None:
        if encoder is None or decoder is None:
            raise TypeError("encoder and decoder are required")

        resolved = self._resolve_backend(
            state,
            backend=backend,
            decay=decay,
            gain=gain,
            dt=dt,
            sensory_indices=sensory_indices,
            model_id=model_id,
        )
        self.backend = resolved
        self.encoder = encoder
        self.decoder = decoder
        self.decay = decay
        self.gain = gain
        self.dt = dt
        self.sensory_indices = sensory_indices
        if isinstance(resolved, InProcessLifBackend):
            self.state: Any = resolved.state
            self.decay = resolved.decay
            self.gain = resolved.gain
            self.dt = resolved.dt
            self.sensory_indices = resolved.sensory_indices
        else:
            inner_state = getattr(resolved, "state", None)
            self.state = inner_state if inner_state is not None else _BackendStateView(resolved)

    @staticmethod
    def _resolve_backend(
        state: BrainState | ModelBackend | None,
        *,
        backend: ModelBackend | None,
        decay: float,
        gain: float,
        dt: float,
        sensory_indices: tuple[int, ...] | None,
        model_id: str | None,
    ) -> ModelBackend:
        if backend is not None:
            if state is not None:
                raise ValueError("pass BrainState or backend, not both")
            return backend
        if isinstance(state, BrainState):
            kwargs: dict[str, Any] = {
                "decay": decay,
                "gain": gain,
                "dt": dt,
                "sensory_indices": sensory_indices,
            }
            if model_id is not None:
                kwargs["model_id"] = model_id
            return InProcessLifBackend(state, **kwargs)
        if state is not None and isinstance(state, ModelBackend):
            return state
        raise TypeError("FlyHarness requires a BrainState or a ModelBackend")

    @property
    def n_neurons(self) -> int:
        return int(self.backend.n_neurons)

    @property
    def model_id(self) -> str:
        return str(self.backend.model_id)

    def reset(self, potentials: np.ndarray | None = None) -> None:
        self.backend.reset(potentials)

    def snapshot(self, *args: Any, **kwargs: Any) -> Any:
        """Checkpoint the docked Model without the loop knowing the engine."""
        return invoke_snapshot(self.backend, *args, **kwargs)

    def restore(self, *args: Any, **kwargs: Any) -> Any:
        """Restore a checkpoint into the docked Model."""
        return invoke_restore(self.backend, *args, **kwargs)

    def step(self, observation: Any) -> StepResult:
        """Encode observation, advance the Model one tick, decode action."""
        input_current = np.asarray(self.encoder.encode(observation), dtype=np.float64)
        n_neurons = self.backend.n_neurons
        if input_current.shape != (n_neurons,):
            raise ValueError(
                "encoder output length must match backend n_neurons "
                f"({n_neurons}), got {input_current.shape}"
            )

        potentials = np.asarray(self.backend.tick(input_current), dtype=np.float64)
        if potentials.shape != (n_neurons,):
            raise ValueError(
                "backend tick output length must match n_neurons "
                f"({n_neurons}), got {potentials.shape}"
            )

        action = self.decoder.decode(potentials)
        return StepResult(
            action=action,
            potentials=potentials.copy(),
            timestamp=float(self.backend.timestamp),
        )
