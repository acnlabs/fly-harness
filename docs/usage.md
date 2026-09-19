# Usage walkthrough

**fly-harness** is the harness half of **Agent = Model + harness**. After
`pip install fly-harness`, prove the kernel is alive, then dock a Model you
already run. The public call is always `FlyHarness.step(obs) -> action`.

This is not a chat shell, web UI, plugin marketplace, or hosted catalog of
brains. The 24-neuron reflex is a **fixture**. The core does **not** bind
flybrain, c302, or any vendor Model. Not a consciousness or upload claim,
not 160k “parameters”, not a Google-hosted tickable brain.

Contract details: [docking.md](docking.md).

<video src="media/visual-demo/fly-walking.mp4" poster="media/visual-demo/01-walking.png" width="800" controls muted loop playsinline></video>

[15 s walk (mp4)](media/visual-demo/fly-walking.mp4) — NeuroMechFly on `FlatTerrain`: walk, then left touch, then right touch.

The 3D walk is **docs/usage display only** — not a kernel feature. Recorded
with the optional FlyGym body. The brain in the default loop is the
**24-neuron LIF fixture**. Legs still run FlyGym's motor **CPG**.
`FlyHarness.step` sits **above** that CPG. Not a 166,700-neuron brain, not
consciousness/upload, not Google-hosted, and not better walking than
FlyGym's own controllers.

## 1. Fixture reflex — just installed / tests / kernel is alive

Default path: toy in-process LIF, **no extras**. Use this after a fresh
`pip install`, in CI, or to prove encode → tick → decode still works.

```bash
pip install fly-harness
fly-harness-check
# same check:
python -m fly_harness
python -m fly_harness.check
```

The command prints `model_id`, `n_neurons`, `timestamp`, and one
obs→action step. Expected fixture id: `fly-harness.in-process-lif`
(24 neurons). Left touch on this toy circuit biases a right turn; do not
treat that as biology.

```python
from fly_harness.check import run_load_check

report = run_load_check()
print(report.model_id, report.n_neurons, report.timestamp, report.action)
```

CI (`pip install -e ".[dev]"` then `pytest`) runs the same load-check
without FlyGym, MCP, flybrain, MaleCNS, or OpenWorm.

## 2. Direct `ModelBackend` — you already run a sim

If you already run a sim (flybrain, c302, Brian2, NEST, your own LIF),
wrap `ModelBackend`. The harness only **encode → tick → decode**. It does
not own that Model.

Required surface: `tick`, `reset`, `n_neurons`, `model_id`, `timestamp`.
Duck typing is enough. You do not subclass a framework.

```python
import numpy as np
from fly_harness import FlyHarness


class IdentityEncoder:
    def encode(self, observation):
        return np.asarray(observation, dtype=np.float64)


class IdentityDecoder:
    def decode(self, potentials):
        return np.asarray(potentials, dtype=np.float64)


class MySimBackend:
    """Your already-running sim. Not a second Model shipped in this repo."""

    def __init__(self, sim, model_id: str = "vendor.mysim") -> None:
        self.sim = sim
        self._model_id = model_id
        self._timestamp = 0.0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def n_neurons(self) -> int:
        return int(self.sim.n)

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def reset(self, potentials=None) -> None:
        if potentials is None:
            self.sim.reset()
        else:
            self.sim.reset(potentials)
        self._timestamp = 0.0

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        out = self.sim.step(input_current)  # your dynamics
        self._timestamp += 1.0
        return np.asarray(out, dtype=np.float64)


# sim is a flybrain / c302 / other tickable process you already run
harness = FlyHarness(
    encoder=IdentityEncoder(),
    decoder=IdentityDecoder(),
    backend=MySimBackend(sim),
)
result = harness.step(observation)
```

Examples and adapters (`examples/flybrain_loop.py`,
`fly_harness.flybrain`, `FakeC302`) are **usage**, not kernel. Core
(`__init__` / `harness` / `backend`) stays model-agnostic and does not
import flybrain or c302.

`examples/flybrain_loop.py` is a runnable Direct dock. Default CI uses
`FakeFlyBrain` (32 neurons). `--real` only if third-party `flybrain` is
installed **and** MaleCNS files are already on disk — it will not download
them. That adapter exists because flybrain’s API is not already a
`ModelBackend`.

## 3. biorouter model ids — same loop, pick an id

Same `FlyHarness.step`. Local stdlib HTTP (`127.0.0.1` by default),
OpenRouter-**shaped** `model` field. Pick `flybrain.malecns` or
`c302.celegans`. Unknown ids are HTTP 404 / `UnknownModelError`.

An extra is glue **only** when a third-party API is not already a `ModelBackend`.
`fly-harness[flybrain]` wraps flybrain so a local
biorouter can list `flybrain.malecns`. There is **no**
`fly-harness[c302]` / `fly-harness[openworm]` extra: listing
`c302.celegans` is deploy a sim + list an id. Default CI registers
`FakeC302` (302 hermaphrodite neurons) so the id exists without NEURON,
Docker, or OpenWorm.

```bash
# local process, not a hosted gateway
biorouter --host 127.0.0.1 --port 8765

# same harness loop; choose one listed id
python examples/biorouter_flybrain_loop.py --url http://127.0.0.1:8765
python examples/biorouter_c302_loop.py --url http://127.0.0.1:8765
```

```python
from fly_harness import BioSimRouter, FlyHarness

router = BioSimRouter(remote_url="http://127.0.0.1:8765", model_id="flybrain.malecns")
# or: model_id="c302.celegans"
harness = FlyHarness(encoder=encoder, decoder=decoder, backend=router)
result = harness.step(observation)
```

In-thread examples (no live daemon) so CI stays green without extras:

```bash
python examples/biorouter_flybrain_loop.py   # fake 32-neuron backend listed as flybrain.malecns
python examples/biorouter_c302_loop.py      # FakeC302, 302 neurons
```

`--real` never downloads connectomes and never starts OpenWorm Docker.
Missing extra or data → omit / 404. This is not a marketplace and not a
second public hosted tickable-brain API.

## Same loop, swap Model

**same loop, swap Model.** The product difference is not a better fly. Keep
the FlyGym body and the `obs → FlyHarness.step → action` loop; change only
the Model.

FlyGym / CPG are **not** the harness kernel. NeuroMechFly's motor CPG lives
in the body extra. `FlyHarness.step` is **brain-level** (encode →
`ModelBackend.tick` → decode), **above** that motor CPG. Swapping the Model
does not replace FlyGym's CPG.

Runnable: `examples/swap_brain.py` (reuses the existing body-loop and
biorouter examples; does not fork the kernel). Backend A is the default toy
in-process LIF (`fly-harness.in-process-lif`). Backend B is listed id
`flybrain.malecns`. Missing `fly-harness[flybrain]` extra or on-disk MaleCNS
→ skip B; will not download; will not pretend **166,700** neurons. Default
CI lists a 32-neuron FakeFlyBrain under that id so the swap path stays green
without extras.

This does not claim better walking than FlyGym's own controllers. Core does
not bind a vendor Model.

```bash
python examples/swap_brain.py
python examples/swap_brain.py --real
```

## What this is not

- Not a chat REPL, web UI, or plugin market
- Not a 140k FlyWire or 166,700-neuron MaleCNS runtime in the core
- Not Google-hosted, not the FlyWire website, not ~160k “parameters”
- Not `caveclient`, not a hosted OpenWorm API
- Not a consciousness or upload claim
