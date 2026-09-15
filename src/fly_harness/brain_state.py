"""BrainState: sparse connectome state with save/load."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

_STATE_VERSION = 1


@dataclass
class BrainState:
    """Mutable neural state: membrane potentials, sparse weights, and clock."""

    potentials: np.ndarray
    weights: sparse.csr_matrix
    timestamp: float = 0.0
    neuron_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.potentials = np.asarray(self.potentials, dtype=np.float64)
        n = self.potentials.shape[0]
        if self.weights.shape != (n, n):
            raise ValueError(
                f"weights shape {self.weights.shape} must match potentials length {n}"
            )
        if not sparse.isspmatrix_csr(self.weights):
            self.weights = self.weights.tocsr()

    @property
    def n_neurons(self) -> int:
        return int(self.potentials.shape[0])

    def copy(self) -> BrainState:
        return BrainState(
            potentials=self.potentials.copy(),
            weights=self.weights.copy(),
            timestamp=self.timestamp,
            neuron_labels=self.neuron_labels,
        )

    def to_dict(self) -> dict[str, Any]:
        coo = self.weights.tocoo()
        return {
            "version": _STATE_VERSION,
            "timestamp": self.timestamp,
            "potentials": self.potentials.tolist(),
            "neuron_labels": list(self.neuron_labels) if self.neuron_labels else None,
            "synapses": {
                "pre": coo.row.tolist(),
                "post": coo.col.tolist(),
                "weight": coo.data.tolist(),
                "shape": list(self.weights.shape),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BrainState:
        version = payload.get("version", 1)
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported BrainState version: {version}")

        potentials = np.asarray(payload["potentials"], dtype=np.float64)
        syn = payload["synapses"]
        n_rows, n_cols = syn["shape"]
        weights = sparse.coo_matrix(
            (syn["weight"], (syn["pre"], syn["post"])),
            shape=(n_rows, n_cols),
        ).tocsr()

        labels = payload.get("neuron_labels")
        neuron_labels = tuple(labels) if labels is not None else None

        return cls(
            potentials=potentials,
            weights=weights,
            timestamp=float(payload.get("timestamp", 0.0)),
            neuron_labels=neuron_labels,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BrainState:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
