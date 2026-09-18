# fly-harness

**v0.5.5** — a sparse neural **harness**, not a fly-brain simulation.

Formula: **Agent = Model + harness**. The harness docks to a **running** sim
through a stable contract (`ModelBackend` only). FlyWire / MaleCNS files are
weight dumps, not that contract. The harness does not dock a specific model.

The public contract is:

1. **`BrainState`** — membrane potentials, sparse synaptic weights (`scipy.sparse` CSR), timestamp, JSON save/load
2. **`FlyHarness.step(obs) -> action`** — encode → **`ModelBackend.tick`** → decode (default backend is the in-process LIF/rate circuit)
3. **`Encoder` / `Decoder`** — `typing.Protocol` contracts (optional `BaseEncoder` / `BaseDecoder` ABCs; no string class-name checks)
4. **`load_connectome`** — load a **circuit subset** from a local sparse edge list (`.npz` / `.csv`) into `BrainState`
5. **`ModelBackend`** — swappable Model port: `tick` (encoded input → neural output), `reset`, `n_neurons`, `model_id`. Optional `snapshot` / `restore` (same family as `reset`; unsupported backends raise, never a silent no-op). See [docs/docking.md](docs/docking.md).

The 24-neuron touch-reflex loop is a **test/demo fixture**. It is not the model, and it is not a scaled-down FlyWire brain.

## What this is not

- **Not** a 140k-neuron FlyWire connectome, upload, or dense 140k×140k matrix
- **Not** an arbitrary-scale whole-brain runtime in the **core** (memory is bounded by the loaded subset). Third-party Models such as [`flybrain`](https://pypi.org/project/flybrain/) (MaleCNS v1.0, **166,700 neurons**, not 160k "parameters") attach through `ModelBackend` — **using** the harness, not a harness feature. A dump you run locally, not a Google-hosted sim or the FlyWire website
- **Not** `caveclient` or a built-in biomechanics engine — FlyGym/NeuroMechFly is an **optional body extra**, not the core product
- **Not** an MCP microkernel or FastAPI service — MCP is an **optional protocol extra**, not the core product
- **Not** OpenRouter-the-company, a marketplace, billing system, hosted cloud gateway, or public bio-sim catalog — `BioSimRouter` is a harness-side **port**; **biorouter** is an optional **local** HTTP process (stdlib, `127.0.0.1` by default)
- **Not** a hosted OpenWorm / c302 API, not `fly-harness[c302]` / `fly-harness[openworm]`, and **not** a new whole-brain PyPI package. Listing `c302.celegans` is usage (deploy a sim + list an id). Default CI uses `FakeC302` (**302** hermaphrodite neurons), not OpenWorm Docker
- **Not** a consciousness/upload claim

## Install

Python 3.10+ with `numpy` and `scipy`. Core install stays numpy/scipy-only.
Just installed (PyPI) — then prove the kernel is alive:

```bash
pip install fly-harness
fly-harness-check
```

From a clone (tests):

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

**biorouter** is optional and **stdlib-only** (no FastAPI). The extra exists so it stays an extension of this repo, not a second product or GitHub repository:

```bash
pip install -e ".[biorouter]"
```

**flybrain** is optional. The extra is an adapter so **biorouter** can list `flybrain.malecns` and the example can use that third-party Model. The harness does not dock a specific model. Core stays model-agnostic and does not import flybrain. Default tests **do not** download `~/fly-data`:

```bash
pip install -e ".[flybrain]"
```

## Usage walkthrough

After install, prove the kernel is alive, then dock a Model you already run.
The public call is always `FlyHarness.step(obs) -> action`. This is not a chat
shell, web UI, or plugin marketplace. Not a consciousness or upload claim.
Longer how-tos: [docs/usage.md](docs/usage.md). Contract:
[docs/docking.md](docs/docking.md).

### 1. Fixture reflex — just installed, no extras

The 24-neuron touch-reflex is a **test fixture**. It proves encode → tick →
decode. It is not a biological circuit.

```bash
pip install fly-harness
fly-harness-check
# or: python -m fly_harness
```

Prints `model_id`, `n_neurons`, `timestamp`, and one obs→action step on the
in-process LIF (`fly-harness.in-process-lif`). CI runs the same check without
extras.

### 2. Direct `ModelBackend` — you already run a sim

The harness core does **not** bind flybrain, c302, or any vendor Model.
If you already have a ticking sim, wrap `ModelBackend` (`tick` / `reset` /
`n_neurons` / `model_id`). The harness only encode → tick → decode.
Examples and adapters (`examples/flybrain_loop.py`, `fly_harness.flybrain`)
are **usage, not kernel**.

```python
from fly_harness import FlyHarness

# sim is your already-running flybrain / c302 / other tickable Model
backend = MySimBackend(sim)  # duck-typed: tick, reset, n_neurons, model_id
harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
result = harness.step(observation)
```

Sketch: [docs/usage.md](docs/usage.md). Runnable Direct example:
`examples/flybrain_loop.py` (FakeFlyBrain in CI; `--real` only if MaleCNS
files are already on disk — never downloads).

### 3. biorouter model ids — same loop, pick an id

Local stdlib HTTP (`127.0.0.1`). Same `FlyHarness.step`. Select
`flybrain.malecns` or `c302.celegans`. Unknown ids are 404 /
`UnknownModelError`. Not a hosted catalog.

An extra is glue **only** when a third-party API is not already a
`ModelBackend`: `fly-harness[flybrain]` wraps flybrain so biorouter can list
`flybrain.malecns`. There is no `fly-harness[c302]` extra.

```bash
biorouter --host 127.0.0.1 --port 8765
python examples/biorouter_flybrain_loop.py --url http://127.0.0.1:8765
python examples/biorouter_c302_loop.py --url http://127.0.0.1:8765
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

`BioSimRouter` is OpenRouter-**shaped**: one entry, a `model` / `model_id` field, unknown ids fail clearly. Optional `remote_url=` forwards that `model` field to **biorouter** (`HttpModelBackend`). This repo does not ship a hosted marketplace.

You can also pass a backend positionally: `FlyHarness(backend, encoder, decoder)`.

## Docking your Model

The harness does not dock a specific vendor. Implement `ModelBackend` (`tick` / `reset` /
`n_neurons` / `model_id` / `timestamp`) and pass it to `FlyHarness.step`. To list that
Model on local **biorouter**, register an OpenRouter-shaped `model` id. Direct dock and
router dock are **usage**, not harness features.

Short guide: [docs/docking.md](docs/docking.md). In-repo stand-in: `FakeDeployedSim`.
Runnable HTTP path: `examples/biorouter_loop.py`. `flybrain.malecns` is **one listed
model** (when extra + on-disk data exist), not “the harness docks flybrain”.
Switching species is the same port: deploy a running sim and list an id.
`c302.celegans` is **one listed** worm Model (default CI: `FakeC302`, **302**
hermaphrodite neurons) — usage, not a harness extra, not a hosted OpenWorm API.

## biorouter

OpenRouter routes existing LLMs. **biorouter** routes existing **deployed biological simulation models**. Same idea (`model` id), different substrate.

This extra is **not** the harness, **not** FlyWire dumps, and **not** a marketplace. It is a local HTTP process (`127.0.0.1` by default) that already-deployed harness clients call: `HttpModelBackend`, `DirectBioSimBackend(url=...)`, `BioSimRouter(remote_url=...)`. Stdlib `http.server` only — no FastAPI, no billing, no cloud gateway.

```bash
pip install -e ".[biorouter]"   # empty extra; core install is enough to run the CLI
biorouter
# or
python -m fly_harness.router --host 127.0.0.1 --port 8765
```

Endpoints (JSON; bodies/query include an OpenRouter-shaped `model` field):

- `POST /tick` — `{"model": "...", "input": [...]}` → neural output
- `POST /reset` — `{"model": "...", "potentials": null | [...]}`
- `GET /status?model=...`

Default in-process registry: the 24-neuron LIF **fixture** (`fly-harness.in-process-lif`), `FakeDeployedSim` ids (`fake.deployed`, `fake.deployed.gain`), and `c302.celegans` (`FakeC302`, **302** hermaphrodite neurons — a listed Model, not a fly-harness extra, not OpenWorm Docker). If `fly-harness[flybrain]` is installed **and** MaleCNS files are already on disk, biorouter also lists `flybrain.malecns` (OpenRouter-shaped provider id). Unknown / missing flybrain → HTTP 404 / skip. Never downloads MaleCNS. Never starts OpenWorm Docker.

```python
from fly_harness import DirectBioSimBackend, FlyHarness

backend = DirectBioSimBackend(
    url="http://127.0.0.1:8765",
    model_id="fake.deployed",
    n_neurons=8,
)
harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
harness.step(observation)
```

`FlyHarness.step(obs) -> action` is unchanged. This extra is not a 140k FlyWire runtime and does not download MaleCNS.

## Example: FlyHarness.step with a third-party flybrain Model

Formula: **Agent = Model + harness**. [`flybrain`](https://pypi.org/project/flybrain/) ([source](https://github.com/alextitonis/fly.ai)) is a **third-party Model** (MaleCNS v1.0 LIF, **166,700 neurons**, 25.6M connections). Google/Janelia released a dump, not a hosted sim. Wiring it is **using** the harness (`ModelBackend`), not a harness feature — the core stays model-agnostic. An optional extra holds the adapter so **biorouter** can list `flybrain.malecns` and this example can run. Not Google-hosted, not 160k parameters, not consciousness.

```python
from fly_harness import FlyHarness
from fly_harness.flybrain import (
    FlyBrainBackend,
    FlyBrainInjectEncoder,
    FlyBrainReadoutDecoder,
)

backend = FlyBrainBackend.from_installed(download=False)  # files must already be on disk
harness = FlyHarness(
    encoder=FlyBrainInjectEncoder(backend.n_neurons, channels={"loom": (0, 1)}),
    decoder=FlyBrainReadoutDecoder(backend.n_neurons),
    backend=backend,
)
result = harness.step({"loom": 0.8})
```

Default example uses an in-process `FakeFlyBrain` (32 neurons). `--real` only runs if `flybrain` imports **and** MaleCNS files are already present; it will not download them:

```bash
python examples/flybrain_loop.py
python examples/flybrain_loop.py --real
# or
python -m fly_harness.demo.flybrain_loop
fly-harness-flybrain-demo
```

## Example: biorouter through FlyHarness.step

OpenRouter routes existing LLMs; **biorouter** routes existing deployed bio-sims. This example starts a **local** stdlib server (or attaches with `--url`), docks `HttpModelBackend` / `DirectBioSimBackend(url=...)`, and calls `FlyHarness.step`. It uses the 8-neuron `FakeDeployedSim` fixture — **not** FlyWire / MaleCNS, **not** a marketplace.

```bash
python examples/biorouter_loop.py
# or, after install:
python -m fly_harness.demo.biorouter_loop
fly-harness-biorouter-demo
# attach to a process you already started:
biorouter --host 127.0.0.1 --port 8765
python examples/biorouter_loop.py --url http://127.0.0.1:8765
```

## Example: biorouter routes flybrain.malecns through FlyHarness.step

OpenRouter lists a provider id; **biorouter** lists `flybrain.malecns` the same way. This example **uses** the harness: local stdlib HTTP → `HttpModelBackend` / `DirectBioSimBackend(url=...)` → `FlyHarness.step({"loom": ...})`. It is not a harness feature — core stays model-agnostic. Default in-thread run registers a fake flybrain-shaped backend (32 neurons) so CI stays green without MaleCNS. `--real` only if `fly-harness[flybrain]` and files are already on disk (never downloads). Missing `flybrain.malecns` → HTTP 404. Not Google-hosted, not 160k parameters, not consciousness.

```bash
python examples/biorouter_flybrain_loop.py
python examples/biorouter_flybrain_loop.py --real
# or, after install:
python -m fly_harness.demo.biorouter_flybrain_loop
fly-harness-biorouter-flybrain-demo
# attach to a process you already started (404 if that process did not list the id):
biorouter --host 127.0.0.1 --port 8765
python examples/biorouter_flybrain_loop.py --url http://127.0.0.1:8765
```

## Example: biorouter routes c302.celegans through FlyHarness.step

Switching species is **usage**, not a harness feature: deploy a running sim + list an OpenRouter-shaped id. **biorouter** lists `c302.celegans` the same way it lists `flybrain.malecns`. There is **no** `fly-harness[c302]` / `fly-harness[openworm]` extra (extra glue is only for a third-party API that is not already `ModelBackend`; that path was rejected for worm). Not a new whole-brain PyPI package, not a hosted OpenWorm API.

Default in-thread run uses `FakeC302` (**302** hermaphrodite neurons) so CI stays green without NEURON / Docker / OpenWorm. `--real` only if third-party `c302` is already importable **and** a tick/reset Model is already running (never downloads connectomes, never starts OpenWorm Docker). Missing / unknown id → HTTP 404 / `UnknownModelError`. Not consciousness.

```bash
python examples/biorouter_c302_loop.py
python examples/biorouter_c302_loop.py --real
# or, after install:
python -m fly_harness.demo.biorouter_c302_loop
fly-harness-biorouter-c302-demo
# attach to a process you already started (404 if that process did not list the id):
biorouter --host 127.0.0.1 --port 8765
python examples/biorouter_c302_loop.py --url http://127.0.0.1:8765
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

## Optional body: FlyGym / NeuroMechFly

This is an **extension**, not the core product. It does not ship a 140k-neuron brain, does not rewrite FlyGym, and does not run MuJoCo unless you install the extra. FlyGym / CPG are **not** the harness kernel: `FlyHarness.step` is brain-level, **above** the body's motor CPG.

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

## Example: FlyGym body through biorouter to flybrain.malecns

FlyGym obs → `FlyHarness.step` → local **biorouter** → listed model id `flybrain.malecns` → decode a FlyGym joint/adhesion action back onto the body. Wiring FlyGym / flybrain is **using** the harness, not a harness feature. Core stays model-agnostic.

Default in-thread run: `StubFlyGymEnv` (no MuJoCo) + `FakeFlyBrain` (32 neurons) listed as `flybrain.malecns`. `--real` only if `fly-harness[flybrain]` and MaleCNS files are already on disk (never downloads). `--try-flygym` attempts a real NeuroMechFly sim and falls back to the stub. Missing / unknown model id → HTTP 404. Not Google-hosted, not 160k parameters, not consciousness.

```bash
python examples/body_loop.py
python examples/body_loop.py --real
python examples/body_loop.py --try-flygym
# or, after install:
python -m fly_harness.demo.body_loop
fly-harness-body-loop-demo
```

## Example: same loop, swap Model

Same FlyGym body. Same `obs → FlyHarness.step → action` loop. Swap only the
Model. This does not claim better walking than FlyGym's own controllers.
FlyGym / CPG are not the harness kernel; `FlyHarness.step` stays brain-level
above the motor CPG.

- **A** — default toy in-process LIF (`fly-harness.in-process-lif`, 24-neuron fixture)
- **B** — biorouter id `flybrain.malecns`

Reuses the body-loop / biorouter examples. Core does not bind a vendor Model.
Default B lists a FakeFlyBrain under that id (32 neurons, **not** 166,700).
`--real` only if `fly-harness[flybrain]` and MaleCNS files are already on disk
(never downloads). Missing extra or data → **skip B** honestly; never a fake
166k brain.

```bash
python examples/swap_brain.py
python examples/swap_brain.py --real
python examples/swap_brain.py --try-flygym
# or, after install:
python -m fly_harness.demo.swap_brain
fly-harness-swap-brain-demo
```

## Tests

GitHub Actions on `main` and pull requests runs `pip install -e ".[dev]"` then `pytest` — no FlyGym/MCP/flybrain extras, no MuJoCo, **no MaleCNS download**, **no OpenWorm Docker**. Skip/mock tests in those extras still pass. The **biorouter** extra is stdlib-only, so its tests run in that same core CI. The load-check (`fly-harness-check`) is core: toy in-process LIF, no extras.

Default local suite does **not** need MuJoCo. The FlyGym smoke test skips unless `flygym` imports and a NeuroMechFly sim can start:

```bash
pytest
# load-check (just-installed path; no extras):
pytest tests/test_check.py
fly-harness-check
python -m fly_harness
# with the FlyGym extra, still skip-friendly if MuJoCo/display is missing:
pytest -q
# FlyGym smoke only:
pytest tests/test_flygym_adapter.py -k smoke
```

Model-backend docking (default LIF, fake deployed sim, router `model_id` switch, unknown-id error) is core and needs no extras:

```bash
pytest tests/test_backend.py tests/test_harness.py
```

The third-party docking guide is markdown only (`docs/docking.md`). A content test checks the `ModelBackend` contract and biorouter listing notes without extras:

```bash
pytest tests/test_docking_guide.py
```

The optional **biorouter** process starts a stdlib server and ticks through `HttpModelBackend` / `FlyHarness` (still no FastAPI, no extras required):

```bash
pytest tests/test_router.py
# after install:
biorouter --help
python -m fly_harness.router --help
```

The biorouter example is in-thread by default (no live `biorouter` daemon). Attaching to a real process is skip-gated on `BIOROUTER_URL`:

```bash
pytest tests/test_biorouter_example.py
python examples/biorouter_loop.py
```

Router tests can register a **fake flybrain-shaped** backend under `flybrain.malecns` (no extra, no download). Default CI without flybrain/data omits that id (HTTP 404). Real flybrain smoke skips unless the extra is installed **and** `~/fly-data` already has files (CI does not download):

```bash
pytest tests/test_router.py tests/test_flybrain_adapter.py
python examples/flybrain_loop.py
# optional, only with extra + on-disk MaleCNS:
# python examples/flybrain_loop.py --real
```

The biorouter → `flybrain.malecns` example is in-thread by default (fake 32-neuron backend listed under that id). Real MaleCNS through HTTP skips unless extra + data are already on disk:

```bash
pytest tests/test_biorouter_flybrain_example.py
python examples/biorouter_flybrain_loop.py
```

Default CI also lists `c302.celegans` as `FakeC302` (302 neurons) so tests can select two model ids (`flybrain.malecns` fake + `c302.celegans` fake) without NEURON / Docker / OpenWorm. Real c302 skips unless a running tick/reset Model is already importable (never downloads, never starts OpenWorm Docker):

```bash
pytest tests/test_biorouter_c302_example.py tests/test_router.py
python examples/biorouter_c302_loop.py
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

The body-loop example is stub body + fake `flybrain.malecns` in-thread (no MuJoCo, no MaleCNS download). Real extras skip unless already present:

```bash
pytest tests/test_body_loop_example.py
python examples/body_loop.py
```

The swap-Model example is the same body + the same loop with two `model_id`s
(toy LIF, then `flybrain.malecns`). Default CI uses the fake listed id (no
MaleCNS download). `--real` without extra/data skips B:

```bash
pytest tests/test_swap_brain_example.py
python examples/swap_brain.py
```

## License

MIT — see [LICENSE](LICENSE).
