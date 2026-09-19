# fly-harness

[English](README.md) | [中文](README.zh-CN.md)

**v0.5.5** — fly-harness sits around a biological sim: obs → encode → Model.tick → decode → action.

Start with the shipped 24-neuron fixture. Same loop if you later plug in a sim you run and swap that Model.

<video src="docs/media/visual-demo/fly-walking.mp4" poster="docs/media/visual-demo/01-walking.png" width="800" controls muted loop playsinline></video>

[15 s walk (mp4)](docs/media/visual-demo/fly-walking.mp4) — NeuroMechFly on `FlatTerrain`: walk, then left touch, then right touch.

<p align="center">
  <img src="docs/media/visual-demo/01-walking.png" alt="WALKING. HUD: brain 24 LIF fixture, legs CPG." width="32%">
  <img src="docs/media/visual-demo/02-touch-left.png" alt="TOUCH LEFT. turn +1. CPG L 1.20 R 0.40." width="32%">
  <img src="docs/media/visual-demo/03-touch-right.png" alt="TOUCH RIGHT. turn -1. CPG L 0.40 R 1.20." width="32%">
</p>

The clip is **usage**, not a kernel feature: 24-neuron LIF fixture at the brain slot, FlyGym CPG still on the legs.

## Install

Python 3.10+ with `numpy` and `scipy`. Fixture included.

```bash
pip install fly-harness
fly-harness-check
```

Same `obs → FlyHarness.step → action` loop if you later swap Model (`examples/swap_brain.py` / `fly-harness-swap-brain-demo`). Optional extras (FlyGym, biorouter, flybrain) live in [docs/usage.md](docs/usage.md). Contract: [docs/docking.md](docs/docking.md).

## What this is not

- Not a 166,700-neuron MaleCNS brain, consciousness, or upload
- Not Google-hosted, not the FlyWire website
- Not better walking than FlyGym's own controllers
- Not a marketplace, hosted catalog, or OpenRouter-the-company

## License

MIT — see [LICENSE](LICENSE).
