# fly-harness

A minimal Python microkernel for stepping observations through a **sparse** neural connectome and reading back actions. Version 0.1 ships a toy 24-neuron reflex loop to prove the architecture—not a full fly brain simulation.

## What this is

- Installable package `fly_harness` with:
  - **`BrainState`** — membrane potentials, sparse synaptic weights (`scipy.sparse`), timestamp, JSON save/load
  - **`FlyHarness.step(obs) -> action`** — one rate-based dynamics step through the connectome
  - **`Encoder` / `Decoder`** — `typing.Protocol` contracts plus optional `BaseEncoder` / `BaseDecoder` ABCs (no string class-name checks)
- Demo sparse connectome (~24 neurons, ~100 synapses) and a touch reflex script

## What this is not

- **Not** a 140k-neuron FlyWire upload or dense 140k×140k weight matrix
- **Not** integrated with `caveclient`, FlyGym, or full biomechanics simulation
- **Not** an MCP server, FastAPI service, or consciousness/upload claim
- **Not** production neuroscience—just a honest v0.1 microkernel skeleton

## Requirements

- Python 3.10+
- `numpy`, `scipy`

## Install

```bash
pip install -e ".[dev]"
```

## Run the reflex demo

```bash
fly-harness-reflex-demo
# or
python -m fly_harness.demo.reflex
```

Example output pattern: left touch drives a right turn (escape away from stimulus); right touch drives a left turn.

## Run tests

```bash
pytest
```

## Quick API sketch

```python
from fly_harness import BrainState, FlyHarness
from fly_harness.demo import ReflexDecoder, ReflexEncoder, build_reflex_connectome
from fly_harness.demo.codec import TouchObservation

state = build_reflex_connectome()
harness = FlyHarness(state, ReflexEncoder(), ReflexDecoder())
result = harness.step(TouchObservation(touch_left=1.0))
print(result.action)  # ReflexAction(turn=1, forward=..., brake=...)

state.save("brain.json")
restored = BrainState.load("brain.json")
```

## License

MIT — see [LICENSE](LICENSE).
