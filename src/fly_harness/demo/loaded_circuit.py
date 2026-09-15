"""Demo: load a sparse connectome fixture and step through FlyHarness."""

from __future__ import annotations

from pathlib import Path

from fly_harness.connectome_loader import load_connectome
from fly_harness.demo.codec import TouchObservation
from fly_harness.harness import FlyHarness
from fly_harness.protocols import BaseDecoder, BaseEncoder

FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "mini_circuit.npz"

SENSORY_LOCAL = (0, 1)
MOTOR_LEFT_LOCAL = (4,)
MOTOR_RIGHT_LOCAL = (5,)


class FixtureEncoder(BaseEncoder):
    def encode(self, observation) -> np.ndarray:
        currents = self._zeros()
        if isinstance(observation, TouchObservation):
            currents[SENSORY_LOCAL[0]] = float(observation.touch_left)
            currents[SENSORY_LOCAL[1]] = float(observation.touch_right)
            return currents
        raise TypeError("expected TouchObservation")


class FixtureDecoder(BaseDecoder):
    def decode(self, potentials: np.ndarray) -> str:
        v = self._validate_potentials(potentials)
        left = float(np.mean(np.clip(v[list(MOTOR_LEFT_LOCAL)], 0.0, None)))
        right = float(np.mean(np.clip(v[list(MOTOR_RIGHT_LOCAL)], 0.0, None)))
        if left > right and left > 0.1:
            return "turn_left"
        if right > left and right > 0.1:
            return "turn_right"
        return "idle"


def main() -> None:
    loaded = load_connectome(FIXTURE)
    harness = FlyHarness(
        state=loaded.state,
        encoder=FixtureEncoder(loaded.n_neurons),
        decoder=FixtureDecoder(loaded.n_neurons),
        sensory_indices=SENSORY_LOCAL,
        decay=0.4,
        gain=0.5,
    )

    print("fly-harness — sparse connectome loader demo")
    print(
        f"loaded {loaded.n_neurons} neurons, "
        f"{loaded.state.weights.nnz} synapses from {FIXTURE.name}"
    )
    print(f"global ids: {loaded.neuron_ids.tolist()}")
    print()

    schedule = [
        ("idle", TouchObservation()),
        ("touch left", TouchObservation(touch_left=1.0)),
        ("touch right", TouchObservation(touch_right=1.0)),
    ]
    for label, obs in schedule:
        result = harness.step(obs)
        print(f"[t={result.timestamp:4.1f}] {label:12s} -> {result.action}")

    print()
    print("Done. Loaded from local NPZ — no dense matrix, no live FlyWire API.")


if __name__ == "__main__":
    main()
