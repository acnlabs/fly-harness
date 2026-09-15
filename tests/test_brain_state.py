"""Tests for BrainState save/load and validation."""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from fly_harness.brain_state import BrainState
from fly_harness.demo.connectome import build_reflex_connectome


def test_reflex_connectome_is_sparse_and_small() -> None:
    state = build_reflex_connectome()
    assert state.n_neurons == 24
    assert state.weights.nnz < 200
    assert sparse.isspmatrix_csr(state.weights)


def test_brain_state_roundtrip_dict() -> None:
    state = build_reflex_connectome()
    state.potentials[0] = 0.42
    state.timestamp = 3.5

    restored = BrainState.from_dict(state.to_dict())
    np.testing.assert_allclose(restored.potentials, state.potentials)
    assert restored.timestamp == state.timestamp
    assert restored.weights.shape == state.weights.shape
    assert restored.weights.nnz == state.weights.nnz
    np.testing.assert_allclose(
        restored.weights.toarray(),
        state.weights.toarray(),
    )


def test_brain_state_save_load(tmp_path: Path) -> None:
    state = build_reflex_connectome()
    state.timestamp = 7.0
    path = tmp_path / "brain.json"
    state.save(path)

    loaded = BrainState.load(path)
    assert loaded.timestamp == 7.0
    assert loaded.n_neurons == state.n_neurons


def test_brain_state_rejects_shape_mismatch() -> None:
    weights = sparse.csr_matrix((3, 3))
    with pytest.raises(ValueError, match="must match"):
        BrainState(potentials=np.zeros(4), weights=weights)
