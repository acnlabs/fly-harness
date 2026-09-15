# fly-harness

**v0.3.0** — a sparse neural **harness**, not a fly-brain simulation.

Formula: **Agent = deployed bio-sim Model + harness**. The harness docks to a
**running** sim through a stable contract. FlyWire / MaleCNS files are weight
dumps, not that contract.

The public contract is:

1. **`BrainState`** — membrane potentials, sparse synaptic weights (`scipy.sparse` CSR), timestamp, JSON save/load
2. **`FlyHarness.step(obs) -> action`** — encode → **`ModelBackend.tick`** → decode (default backend is the in-process LIF/rate circuit)
3. **`Encoder` / `Decoder`** — `typing.Protocol` contracts (optional `BaseEncoder` / `BaseDecoder` ABCs; no string class-name checks)
4. **`load_connectome`** — load a **circuit subset** from a local sparse edge list (`.npz` / `.csv`) into `BrainState`
5. **`ModelBackend`** — swappable Model port: `tick` (encoded input → neural output), `reset`, `n_neurons`, `model_id`

The 24-neuron touch-reflex loop is a **test/demo fixture**. It is not the model, and it is not a scaled-down FlyWire brain.

## What this is not

- **Not** a 140k-neuron FlyWire connectome, upload, or dense 140k×140k matrix
- **Not** an arbitrary-scale whole-brain runtime (memory is bounded by the loaded subset)
- **Not** `caveclient` or a built-in biomechanics engine — FlyGym/NeuroMechFly is an **optional body extra**, not the core product
- **Not** an MCP microkernel or FastAPI service — MCP is an **optional protocol extra**, not the core product
- **Not** a public bio-sim OpenRouter, marketplace, billing system, or multi-tenant gateway — the router is a harness-side **port**
- **Not** a consciousness/upload claim

## Install

Python 3.10+ with `numpy` and `scipy`. Core install stays numpy/scipy-only:

```bash
pip install -e ".[dev]"
```

FlyGym is optional:

```bash
pip install -e ".[flygym]"
```

MCP is optional (stdio server via FastMCP):

```bash
pip install -e ".[mcp]"
```

## The step loop

```python
from fly_harness import BrainState, FlyHarness
from fly_harness.protocols import Decoder, Encoder

harness = FlyHarness(state, encoder, decoder)
result = harness.step(observation)
action = result.action
```

`step` encodes the observation into a length-`n_neurons` current vector, advances the docked **Model** one tick, and decodes an action from the neural output.

Default construction still wraps the in-process LIF/rate dynamics (`InProcessLifBackend`). `FlyHarness.step(obs) -> action` is the same public call.

`BrainState` can be snapshotted independently of the harness:

```python
state.save("brain.json")
restored = BrainState.load("brain.json")
```

## Docking modes

Two ways to attach a running Model. This package is the **port**, not a hosted catalog of sims.

### 1. Direct — one backend per vendor / deployed sim

```python
from fly_harness import DirectBioSimBackend, FlyHarness

backend = DirectBioSimBackend(n_neurons=8, model_id="vendor.fake")
harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
result = harness.step(observation)
```

`DirectBioSimBackend` docks to **one** running sim. The in-process `FakeDeployedSim` is enough for tests. A thin HTTP client is available if you already host a sim (`url=...`, bodies include a `model` field). There are no FlyWire / CAVE / neuPrint clients and no 140k dense matrices.

### 2. Router — one entry, select by model id

```python
from fly_harness import BioSimRouter, DirectBioSimBackend, FlyHarness, UnknownModelError

router = BioSimRouter(
    {
        "vendor.a": DirectBioSimBackend(n_neurons=8, model_id="vendor.a"),
        "vendor.b": DirectBioSimBackend(n_neurons=8, model_id="vendor.b"),
    },
    model_id="vendor.a",
)
harness = FlyHarness(encoder=encoder, decoder=decoder, backend=router)
harness.step(observation)
router.use("vendor.b")
harness.step(observation)

# unknown model_id raises UnknownModelError (this is a port, not a marketplace)
```

`BioSimRouter` is OpenRouter-**shaped**: one entry, a `model` / `model_id` field, unknown ids fail clearly. Optional `remote_url=` forwards that `model` field to an HTTP router **you** run later (`HttpModelBackend`). This repo does not ship a hosted marketplace.

You can also pass a backend positionally: `FlyHarness(backend, encoder, decoder)`.

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

## Optional body: FlyGym / NeuroMechFly

This is an **extension**, not the core product. It does not ship a 140k-neuron brain, does not rewrite FlyGym, and does not run MuJoCo unless you install the extra.

`FlyGymEncoder` / `FlyGymDecoder` map NeuroMechFly observations (joint angles, contact forces) and actions (joint targets, per-leg adhesion, optional muscle/tendon commands) onto `FlyHarness.step`. They work on dicts — no MuJoCo import. `FlyGymHarnessEnv` is a thin wrapper around an env **you** construct with FlyGym.

```python
from fly_harness import FlyHarness
from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome
from fly_harness.flygym import FlyGymDecoder, FlyGymEncoder, FlyGymHarnessEnv

state = build_reflex_connectome()  # fixture circuit, not FlyWire
harness = FlyHarness(
    state,
    FlyGymEncoder(state.n_neurons),
    FlyGymDecoder(state.n_neurons),
    sensory_indices=SENSORY_INDICES,
)
# env = your FlyGym / NeuroMechFly simulation (not constructed here)
body = FlyGymHarnessEnv(env, harness)
obs, info = body.reset()
result = body.step()
# result.action.as_env_dict() -> {"joints": ..., "adhesion": ...}
```

Gymnasium FlyGym (`flygym-gymnasium`, installed by the extra; import name `flygym` or `flygym_gymnasium`) takes `env.step({"joints", "adhesion"})`. FlyGym 2.x `Simulation` is duck-typed via `set_actuator_inputs` / `set_leg_adhesion_states` when you pass `actuator_type`.

## Optional protocol: MCP

This is an **extension pack**, not the microkernel. `BrainState` / `FlyHarness` stay numpy/scipy-only and do not import MCP. The default server steps the 24-neuron reflex **fixture**, not a 140k FlyWire brain.

```bash
pip install -e ".[mcp]"
# stdio (default FastMCP transport) — point an MCP host at this command:
fly-harness-mcp
# or
python -m fly_harness.mcp
```

Tools (no live client required to unit-test the handlers):

- `harness_step(touch_left, touch_right)` — one `FlyHarness.step`
- `harness_reset()` — zero potentials and the clock
- `harness_status()` — neuron count, timestamp, package version

Handlers are importable without FastMCP:

```python
from fly_harness.mcp import HarnessSession

session = HarnessSession()
print(session.step(touch_left=1.0)["action"])
```

## Example: protocol + body through the harness

This proves optional **MCP** and **FlyGym** extras can attach to the same `FlyHarness.step`. It uses the 24-neuron reflex **fixture**. It is **not** a 140k FlyWire or ~166k MaleCNS upload, and it does not rewrite FlyGym or the MCP pack.

```bash
python examples/mcp_flygym_loop.py
# or, after install:
python -m fly_harness.demo.composed
fly-harness-composed-demo
```

Default run uses an in-process body stub (no MuJoCo). `--try-flygym` attempts a real NeuroMechFly sim and falls back to the stub. `--serve` starts the optional FastMCP stdio server on that composed session (`pip install -e ".[mcp]"`):

```bash
python examples/mcp_flygym_loop.py --serve
```

An MCP host that calls `harness_step(touch_left, touch_right)` therefore steps the harness and applies the decoded joint/adhesion command to the body.

## Tests

GitHub Actions on `main` and pull requests runs `pip install -e ".[dev]"` then `pytest` — no FlyGym/MCP extras, no MuJoCo. Skip/mock tests in those extras still pass.

Default local suite does **not** need MuJoCo. The FlyGym smoke test skips unless `flygym` imports and a NeuroMechFly sim can start:

```bash
pytest
# with the FlyGym extra, still skip-friendly if MuJoCo/display is missing:
pytest -q
# FlyGym smoke only:
pytest tests/test_flygym_adapter.py -k smoke
```

Model-backend docking (default LIF, fake deployed sim, router `model_id` switch, unknown-id error) is core and needs no extras:

```bash
pytest tests/test_backend.py tests/test_harness.py
```

MCP tests mock FastMCP and do **not** start a live MCP client. They pass without `fly-harness[mcp]`. With the extra, a factory smoke checks FastMCP constructs (still no stdio client):

```bash
pip install -e ".[mcp]"
pytest tests/test_mcp_extension.py
```

The composed MCP+FlyGym example is also mock/skip: no MuJoCo and no live MCP client.

```bash
pytest tests/test_composed_example.py
python examples/mcp_flygym_loop.py
```

## License

MIT — see [LICENSE](LICENSE).
