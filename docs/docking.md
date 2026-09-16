# Docking a deployed bio-sim

**fly-harness** is a port, not a catalog of brains. Formula: **Agent = Model + harness**.
You bring a **running** sim. Direct dock and router dock are **usage**, not harness
features. The core stays a model-agnostic `ModelBackend` contract. This file is not
a FlyWire / MaleCNS runtime, not a marketplace, and not a consciousness or upload
claim.

`flybrain.malecns` is **one listed model** on local biorouter when that extra and
on-disk data are present. It is not “the harness docks flybrain”.

## 1. Implement `ModelBackend`

Structural typing (`typing.Protocol`). You do not subclass a framework. Required surface:

| Member | Role |
| --- | --- |
| `model_id` | Stable id (router key / JSON `model` field) |
| `n_neurons` | Length of encoded input and neural output |
| `timestamp` | Clock after the last `tick` or `reset` |
| `reset(potentials=None)` | Clear or restore neural state; rewind the clock |
| `tick(input_current)` | One step: length-`n_neurons` current → neural output |

In-repo stand-in (tests, not a connectome dump): `FakeDeployedSim` in
`src/fly_harness/backend.py`. Contract tests: `tests/test_backend.py`.

```python
import numpy as np
from fly_harness import FlyHarness
from fly_harness.protocols import ModelBackend  # optional; duck typing is enough


class VendorSim:
    """Your deployed bio-sim. This sketch is not a second Model in the repo."""

    def __init__(self, n: int, model_id: str = "vendor.sim") -> None:
        self._model_id = model_id
        self._n_neurons = n
        self._timestamp = 0.0
        self.potentials = np.zeros(n, dtype=np.float64)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def reset(self, potentials: np.ndarray | None = None) -> None:
        self.potentials[:] = 0.0 if potentials is None else np.asarray(potentials, dtype=np.float64)
        self._timestamp = 0.0

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        current = np.asarray(input_current, dtype=np.float64)
        if current.shape != (self._n_neurons,):
            raise ValueError("encoded input length must match n_neurons")
        self.potentials = current  # replace with your dynamics
        self._timestamp += 1.0
        return self.potentials.copy()


class IdentityEncoder:
    def encode(self, observation):
        return np.asarray(observation, dtype=np.float64)


class IdentityDecoder:
    def decode(self, potentials):
        return np.asarray(potentials, dtype=np.float64)


backend = VendorSim(8)
assert isinstance(backend, ModelBackend)
harness = FlyHarness(encoder=IdentityEncoder(), decoder=IdentityDecoder(), backend=backend)
result = harness.step(np.ones(8))
action = result.action
```

`Encoder` / `Decoder` must emit and consume length-`n_neurons` vectors. `FlyHarness.step(obs) -> action`
is encode → `tick` → decode. The harness does not load FlyWire or MaleCNS dumps here.

## 2. Direct dock

Pass your backend into `FlyHarness` (`backend=` or positionally). Helpers:

- `DirectBioSimBackend(sim=...)` — wrap an in-process object
- `DirectBioSimBackend(url=..., model_id=..., n_neurons=...)` — thin HTTP client if you already host `POST /tick`, `POST /reset`, `GET /status` with a `model` field
- `FakeDeployedSim` — local stand-in so you can write tests without your sim

There are no FlyWire / CAVE / neuPrint clients in core.

## 3. List the model on biorouter

OpenRouter lists LLM provider ids. **biorouter** lists **deployed bio-sim** ids the same way
(`model` field), on a **local** stdlib process (`127.0.0.1` by default). Not
OpenRouter-the-company, not billing, not a hosted gateway.

Register your backend under a `model` id and serve that registry. Unknown ids are HTTP 404
→ harness `UnknownModelError`.

```python
from fly_harness.router.registry import create_default_backends, create_registry
from fly_harness.router.server import serve

backends = create_default_backends()  # fixtures; optional flybrain.malecns if extra+data exist
backends["vendor.sim"] = VendorSim(8, model_id="vendor.sim")
serve(registry=create_registry(backends))  # POST /tick {"model": "vendor.sim", "input": [...]}
```

The `biorouter` CLI uses the default registry only. Listing a third-party sim is
**process code** (as above), not a plugin marketplace. Clients keep calling
`HttpModelBackend` / `DirectBioSimBackend(url=...)` / `BioSimRouter(remote_url=...)`.

Default CLI ids: `fly-harness.in-process-lif`, `fake.deployed`, `fake.deployed.gain`.
`flybrain.malecns` is listed only when `fly-harness[flybrain]` **and** MaleCNS files are
already on disk. Missing extra or data → omit / 404. This extra will not download MaleCNS.

## Existing examples

Do not treat these as a second Model or a flybrain product surface. They show the port:

| Path | What it shows |
| --- | --- |
| `tests/test_backend.py` | `FakeDeployedSim` / `DirectBioSimBackend` / `BioSimRouter` through `FlyHarness.step` |
| `examples/biorouter_loop.py` | Local HTTP → fixture ids (`fake.deployed`) |
| `examples/biorouter_flybrain_loop.py` | Same HTTP path; `flybrain.malecns` as **one listed** id (fake in CI) |
| `examples/flybrain_loop.py` | Using a third-party Model directly (not a harness feature) |
| `examples/body_loop.py` | FlyGym-shaped obs through the same port to a listed id |

## What this is not

- Not a 140k FlyWire or ~166k MaleCNS runtime in the core
- Not Google-hosted, not the FlyWire website, not ~160k “parameters”
- Not caveclient, not a biomechanics engine (FlyGym is an optional extra)
- Not FastAPI, not a public catalog
- Not a consciousness or upload claim
