"""Tests for sparse connectome loading."""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from fly_harness import FlyHarness
from fly_harness.connectome_loader import ConnectomeLoadResult, load_connectome
from fly_harness.demo.codec import ReflexDecoder, ReflexEncoder, TouchObservation
from fly_harness.demo.connectome import build_reflex_connectome
from fly_harness.demo.reflex import run_reflex_loop
from fly_harness.protocols import BaseDecoder, BaseEncoder

FIXTURES = Path(__file__).parent / "fixtures"
NPZ_FIXTURE = FIXTURES / "mini_circuit.npz"
CSV_FIXTURE = FIXTURES / "mini_circuit.csv"
IDS_FIXTURE = FIXTURES / "mini_circuit_neuron_ids.txt"

# Local indices after loading SUBSET_IDS from the NPZ fixture.
SENSORY_LOCAL = (0, 1)
MOTOR_LEFT_LOCAL = (4,)
MOTOR_RIGHT_LOCAL = (5,)


class MiniCircuitEncoder(BaseEncoder):
    def encode(self, observation) -> np.ndarray:
        currents = self._zeros()
        if isinstance(observation, TouchObservation):
            currents[SENSORY_LOCAL[0]] = float(observation.touch_left)
            currents[SENSORY_LOCAL[1]] = float(observation.touch_right)
            return currents
        raise TypeError("expected TouchObservation")


class MiniCircuitDecoder(BaseDecoder):
    def decode(self, potentials: np.ndarray) -> str:
        v = self._validate_potentials(potentials)
        left = float(np.mean(np.clip(v[list(MOTOR_LEFT_LOCAL)], 0.0, None)))
        right = float(np.mean(np.clip(v[list(MOTOR_RIGHT_LOCAL)], 0.0, None)))
        if left > right and left > 0.1:
            return "turn_left"
        if right > left and right > 0.1:
            return "turn_right"
        return "idle"


def test_load_npz_fixture_maps_to_brain_state() -> None:
    result = load_connectome(NPZ_FIXTURE)
    assert isinstance(result, ConnectomeLoadResult)
    assert result.n_neurons == 6
    assert result.state.weights.shape == (6, 6)
    assert sparse.isspmatrix_csr(result.state.weights)
    assert result.state.weights.nnz == 5
    assert result.neuron_ids.tolist() == [581001, 581002, 581101, 581102, 581201, 581301]


def test_load_csv_with_explicit_subset() -> None:
    subset = [581001, 581002, 581101, 581102, 581201, 581302]
    result = load_connectome(CSV_FIXTURE, neuron_ids=subset)
    assert result.n_neurons == 6
    assert result.state.weights.nnz == 5
    assert 581303 not in result.id_to_index


def test_load_neuron_ids_from_text_file() -> None:
    result = load_connectome(CSV_FIXTURE, neuron_ids_file=IDS_FIXTURE)
    assert result.n_neurons == 6
    assert result.neuron_ids.tolist() == load_connectome(NPZ_FIXTURE).neuron_ids.tolist()


def test_subset_bounds_memory_not_full_brain() -> None:
    tiny = [581001, 581101, 581301]
    result = load_connectome(NPZ_FIXTURE, neuron_ids=tiny)
    assert result.n_neurons == 3
    assert result.state.weights.shape == (3, 3)
    assert result.state.weights.nnz == 3


def test_no_edges_after_filter_raises() -> None:
    with pytest.raises(ValueError, match="no synapses remain"):
        load_connectome(NPZ_FIXTURE, neuron_ids=[999999])


def test_fixture_runs_harness_step() -> None:
    loaded = load_connectome(NPZ_FIXTURE)
    harness = FlyHarness(
        state=loaded.state,
        encoder=MiniCircuitEncoder(loaded.n_neurons),
        decoder=MiniCircuitDecoder(loaded.n_neurons),
        sensory_indices=SENSORY_LOCAL,
        decay=0.4,
        gain=0.5,
    )
    harness.step(TouchObservation())
    result = harness.step(TouchObservation(touch_left=1.0))
    assert result.action == "turn_right"
    assert result.timestamp == 2.0


def test_reflex_demo_still_works() -> None:
    state = build_reflex_connectome()
    assert state.n_neurons == 24
    actions = run_reflex_loop(steps=6)
    assert actions[2].turn == 1
    assert actions[5].turn == -1


def test_reflex_codec_still_compatible_with_reflex_connectome() -> None:
    harness = FlyHarness(
        state=build_reflex_connectome(),
        encoder=ReflexEncoder(),
        decoder=ReflexDecoder(),
    )
    result = harness.step(TouchObservation(touch_left=1.0))
    assert hasattr(result.action, "turn")
