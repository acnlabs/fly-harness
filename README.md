# fly-harness

**v0.1.1** — a sparse neural **harness**, not a fly-brain simulation.

The public contract is:

1. **`BrainState`** — membrane potentials, sparse synaptic weights (`scipy.sparse` CSR), timestamp, JSON save/load
2. **`FlyHarness.step(obs) -> action`** — one rate-based dynamics step through that connectome
3. **`Encoder` / `Decoder`** — `typing.Protocol` contracts (optional `BaseEncoder` / `BaseDecoder` ABCs; no string class-name checks)
4. **`load_connectome`** — load a **circuit subset** from a local sparse edge list (`.npz` / `.csv`) into `BrainState`

The 24-neuron touch-reflex loop is a **test/demo fixture**. It is not the model, and it is not a scaled-down FlyWire brain.

## What this is not

- **Not** a 140k-neuron FlyWire connectome, upload, or dense 140k×140k matrix
- **Not** an arbitrary-scale whole-brain runtime (memory is bounded by the loaded subset)
- **Not** FlyGym, `caveclient`, or a biomechanics simulator
- **Not** an MCP server, FastAPI service, or consciousness/upload claim

## Install

Python 3.10+ with `numpy` and `scipy`.

```bash
pip install -e ".[dev]"
```

## The step loop

```python
from fly_harness import BrainState, FlyHarness
from fly_harness.protocols import Decoder, Encoder

harness = FlyHarness(state, encoder, decoder)
result = harness.step(observation)
action = result.action
```

`step` encodes the observation into a length-`n_neurons` current vector, advances sparse rate dynamics once, and decodes an action from the updated potentials.

`BrainState` can be snapshotted independently of the harness:

```python
state.save("brain.json")
restored = BrainState.load("brain.json")
```

## Load a sparse connectome

`load_connectome` maps global neuron ids (any integers) onto contiguous local indices and builds a CSR weight matrix of shape `(n_subset, n_subset)`.

```python
from fly_harness import FlyHarness, load_connectome

loaded = load_connectome("tests/fixtures/mini_circuit.npz")
# loaded.state.weights is scipy.sparse CSR, shape (n_subset, n_subset)
# loaded.neuron_ids maps local index -> global neuron id
# loaded.id_to_index maps global id -> local index

harness = FlyHarness(loaded.state, encoder=..., decoder=...)
result = harness.step(observation)
```

```bash
fly-harness-loader-demo
# or
python -m fly_harness.demo.loaded_circuit
```

Supported files:

- **NPZ** — arrays `pre`, `post`, `weight` (int ids + float weights); optional `neuron_ids` for the circuit subset
- **CSV** — header `pre_id,post_id,weight`

Pass an explicit subset with `neuron_ids=[...]` or `neuron_ids_file="ids.txt"` (one id per line). Only synapses whose pre and post are both in the subset are kept.

## Fixture: 24-neuron reflex

Shipped only so the contract is executable without an external connectome file. Left touch biases a right turn (and the reverse). Do not treat this as a biological circuit.

```bash
fly-harness-reflex-demo
# or
python -m fly_harness.demo.reflex
```

```python
from fly_harness import FlyHarness
from fly_harness.demo import ReflexDecoder, ReflexEncoder, build_reflex_connectome
from fly_harness.demo.codec import TouchObservation

state = build_reflex_connectome()
harness = FlyHarness(state, ReflexEncoder(), ReflexDecoder())
result = harness.step(TouchObservation(touch_left=1.0))
print(result.action)  # ReflexAction(turn=1, forward=..., brake=...)
```

## Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
