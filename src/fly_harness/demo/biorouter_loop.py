"""Runnable example: local biorouter process → HttpModelBackend → FlyHarness.step.

OpenRouter routes existing LLMs; biorouter routes existing **deployed
biological simulation models**. Same ``model`` id idea, different substrate.

Default run starts an in-thread stdlib server (no separate ``biorouter``
daemon, no FastAPI, no FlyWire/MaleCNS download). Pass ``--url`` to attach
to a process you already started with ``biorouter``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from fly_harness.backend import BioSimRouter, DirectBioSimBackend, UnknownModelError
from fly_harness.harness import FlyHarness
from fly_harness.router.registry import (
    DEFAULT_FAKE_GAIN_MODEL_ID,
    DEFAULT_FAKE_MODEL_ID,
    DEFAULT_FAKE_N_NEURONS,
)
from fly_harness.router.server import running_router

FIXTURE_FAKE_N_NEURONS = DEFAULT_FAKE_N_NEURONS  # 8 — not 140k / 166k
_BANNER = (
    "biorouter example: OpenRouter routes existing LLMs; biorouter routes "
    "existing deployed biological simulation models. Local stdlib HTTP → "
    f"HttpModelBackend → FlyHarness.step on an {FIXTURE_FAKE_N_NEURONS}-neuron "
    "FakeDeployedSim (not the harness, not FlyWire dumps, not a marketplace)."
)


class VectorEncoder:
    def encode(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float64)


class VectorDecoder:
    def decode(self, potentials: np.ndarray) -> np.ndarray:
        return np.asarray(potentials, dtype=np.float64)


@dataclass
class BiorouterLoop:
    """Harness docked to a local (or attached) biorouter over HTTP."""

    url: str
    router: BioSimRouter
    harness: FlyHarness

    @property
    def n_neurons(self) -> int:
        return int(self.harness.n_neurons)

    @property
    def model_id(self) -> str:
        return str(self.harness.model_id)


def build_loop(
    url: str,
    *,
    model_id: str = DEFAULT_FAKE_MODEL_ID,
    n_neurons: int = FIXTURE_FAKE_N_NEURONS,
) -> BiorouterLoop:
    """Wire FlyHarness.step to biorouter via Direct + BioSimRouter(remote_url=)."""
    direct = DirectBioSimBackend(url=url, model_id=model_id, n_neurons=n_neurons)
    router = BioSimRouter(remote_url=url, model_id=model_id)
    harness = FlyHarness(
        encoder=VectorEncoder(),
        decoder=VectorDecoder(),
        backend=direct,
    )
    return BiorouterLoop(url=url, router=router, harness=harness)


def iter_inprocess_loop() -> Iterator[BiorouterLoop]:
    """Start a local stdlib biorouter (no CLI subprocess) and yield a loop."""
    with running_router() as (url, _server):
        yield build_loop(url)


def run_scripted_loop(
    loop: BiorouterLoop,
    *,
    steps: int = 2,
    switch_to: str = DEFAULT_FAKE_GAIN_MODEL_ID,
) -> list[dict[str, Any]]:
    """Tick through HttpModelBackend, switch model id, then hit an unknown id."""
    stimulus = np.ones(loop.n_neurons, dtype=np.float64)
    traces: list[dict[str, Any]] = []
    loop.harness.reset()
    for i in range(max(1, steps)):
        result = loop.harness.step(stimulus)
        traces.append(
            {
                "phase": "direct",
                "step": i,
                "model_id": loop.harness.model_id,
                "n_neurons": loop.n_neurons,
                "timestamp": float(result.timestamp),
                "potentials": result.potentials.tolist(),
            }
        )

    switched = BioSimRouter(remote_url=loop.url, model_id=switch_to)
    switch_harness = FlyHarness(
        encoder=VectorEncoder(),
        decoder=VectorDecoder(),
        backend=switched,
    )
    switched_out = switch_harness.step(stimulus)
    traces.append(
        {
            "phase": "switch",
            "step": 0,
            "model_id": switch_harness.model_id,
            "n_neurons": switch_harness.n_neurons,
            "timestamp": float(switched_out.timestamp),
            "potentials": switched_out.potentials.tolist(),
        }
    )

    unknown = BioSimRouter(remote_url=loop.url)
    forwarded = unknown.select("ghost.sim")
    try:
        forwarded.tick(np.zeros(loop.n_neurons, dtype=np.float64))
        traces.append({"phase": "unknown", "error": None})
    except UnknownModelError as exc:
        traces.append(
            {
                "phase": "unknown",
                "error": type(exc).__name__,
                "model_id": exc.model_id,
            }
        )
    return traces


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=_BANNER)
    parser.add_argument(
        "--url",
        default=None,
        help="attach to an already-running biorouter (default: in-thread stdlib server)",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--model",
        default=DEFAULT_FAKE_MODEL_ID,
        help="OpenRouter-shaped model id (default: fake.deployed)",
    )
    args = parser.parse_args(argv)

    print(_BANNER)

    def _run(loop: BiorouterLoop) -> None:
        print(
            f"url={loop.url} n_neurons={loop.n_neurons} model={loop.model_id} "
            "(fixture FakeDeployedSim; not MaleCNS 166k)"
        )
        traces = run_scripted_loop(loop, steps=args.steps)
        for row in traces:
            if row["phase"] == "unknown":
                print(
                    f"unknown model: error={row.get('error')} "
                    f"model_id={row.get('model_id')}"
                )
                continue
            pots = np.asarray(row["potentials"])
            print(
                f"{row['phase']} step {row['step']}: model={row['model_id']} "
                f"n_neurons={row['n_neurons']} timestamp={row['timestamp']} "
                f"potentials[:3]={pots[:3]}"
            )

    if args.url:
        _run(build_loop(args.url, model_id=args.model))
        return

    with running_router() as (url, _server):
        _run(build_loop(url, model_id=args.model))


if __name__ == "__main__":
    main()
