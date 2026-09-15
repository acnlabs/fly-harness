"""Encoder, decoder, and ModelBackend contracts for the fly harness microkernel."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    """Maps environment observations to input currents for sensory neurons."""

    def encode(self, observation: Any) -> np.ndarray:
        """Return a length-n_neurons vector of input currents."""
        ...


@runtime_checkable
class Decoder(Protocol):
    """Maps neuron potentials to environment actions."""

    def decode(self, potentials: np.ndarray) -> Any:
        """Return an action derived from the current neuron potentials."""
        ...


@runtime_checkable
class ModelBackend(Protocol):
    """A running bio-sim Model the harness docks to.

    ``tick`` consumes an encoded current vector and returns neural output
    (typically membrane potentials). The harness does not load FlyWire /
    MaleCNS dumps here — those are weight files, not a deployed Model.
    """

    @property
    def model_id(self) -> str:
        """Stable id for this deployed Model (router key / ``model`` field)."""
        ...

    @property
    def n_neurons(self) -> int:
        """Length of the encoded input and neural output vectors."""
        ...

    @property
    def timestamp(self) -> float:
        """Backend clock after the last tick or reset."""
        ...

    def reset(self, potentials: np.ndarray | None = None) -> None:
        """Clear or replace neural state and rewind the clock."""
        ...

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        """Advance one neural tick: encoded input -> neural output."""
        ...


class BaseEncoder(ABC):
    """Optional ABC for encoders that need shared setup or validation."""

    def __init__(self, n_neurons: int) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        self.n_neurons = n_neurons

    @abstractmethod
    def encode(self, observation: Any) -> np.ndarray:
        """Return a length-n_neurons vector of input currents."""

    def _zeros(self) -> np.ndarray:
        return np.zeros(self.n_neurons, dtype=np.float64)


class BaseDecoder(ABC):
    """Optional ABC for decoders that need shared setup or validation."""

    def __init__(self, n_neurons: int) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        self.n_neurons = n_neurons

    @abstractmethod
    def decode(self, potentials: np.ndarray) -> Any:
        """Return an action derived from the current neuron potentials."""

    def _validate_potentials(self, potentials: np.ndarray) -> np.ndarray:
        arr = np.asarray(potentials, dtype=np.float64)
        if arr.shape != (self.n_neurons,):
            raise ValueError(
                f"expected potentials shape ({self.n_neurons},), got {arr.shape}"
            )
        return arr
