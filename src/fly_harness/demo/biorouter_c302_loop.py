"""Example: FlyHarness.step via local biorouter to model id c302.celegans.

Wiring a worm sim is using the harness, not a harness feature. Switching
species = deploy a running sim + list an id. OpenRouter lists a provider;
biorouter lists ``c302.celegans`` the same way.

Default in-thread run uses the default registry's FakeC302 (302 hermaphrodite
neurons, no NEURON / Docker / OpenWorm). Pass ``--real`` only when third-party
``c302`` is already importable *and* a tick/reset Model is already running —
this repo will not download connectomes and will not start OpenWorm Docker.
``--url`` attaches to a process you already started; missing
``c302.celegans`` is HTTP 404.

    python examples/biorouter_c302_loop.py
    python examples/biorouter_c302_loop.py --real
    python -m fly_harness.demo.biorouter_c302_loop
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np

from fly_harness.backend import BioSimRouter, DirectBioSimBackend, UnknownModelError
from fly_harness.c302 import (
    C302_MODEL_ID,
    C302_N_NEURONS,
    FakeC302,
    c302_available,
    running_c302_backend,
)
from fly_harness.harness import FlyHarness
from fly_harness.http_backend import HttpModelBackend
from fly_harness.router import create_default_backends, create_registry
from fly_harness.router.server import running_router

_BANNER = (
    "biorouter → c302.celegans example: obs → encode → Model.tick → decode → action. "
    "Switching species is deploy a sim + list an id. Worm is usage, not a "
    "harness feature. OpenRouter lists a provider id; biorouter lists "
    "c302.celegans. FlyHarness.step goes through local stdlib HTTP. "
    "Default is FakeC302 (302 hermaphrodite neurons, not OpenWorm Docker, "
    "not a hosted OpenWorm API). Not a fly-harness[c302] extra, not a new "
    "whole-brain PyPI package, not a consciousness or upload claim. Never "
    "downloads connectomes. Never starts OpenWorm Docker."
)


class TouchEncoder:
    """Map a small touch observation onto a length-n_neurons current vector."""

    def __init__(self, n_neurons: int, *, n_touch: int = 4) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        self.n_neurons = int(n_neurons)
        self.n_touch = max(1, min(int(n_touch), self.n_neurons))

    def encode(self, observation: Any) -> np.ndarray:
        currents = np.zeros(self.n_neurons, dtype=np.float64)
        if isinstance(observation, np.ndarray):
            arr = np.asarray(observation, dtype=np.float64).reshape(-1)
            if arr.shape != (self.n_neurons,):
                raise ValueError(
                    f"expected currents shape ({self.n_neurons},), got {arr.shape}"
                )
            return arr
        if not isinstance(observation, Mapping):
            raise TypeError("TouchEncoder expects a current vector or a dict")
        amount = float(observation.get("touch", 0.0))
        currents[: self.n_touch] = amount
        return currents


class MotorDecoder:
    """Mean voltage over a tail of motor-like indices. Not OpenWorm cell names."""

    def __init__(self, n_neurons: int, *, n_motor: int = 8) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        self.n_neurons = int(n_neurons)
        self.n_motor = max(1, min(int(n_motor), self.n_neurons))

    def decode(self, potentials: np.ndarray) -> dict[str, float | int]:
        v = np.asarray(potentials, dtype=np.float64).reshape(-1)
        if v.shape != (self.n_neurons,):
            raise ValueError(f"expected potentials shape ({self.n_neurons},), got {v.shape}")
        pool = v[-self.n_motor :]
        return {
            "n_neurons": self.n_neurons,
            "motor_mean": float(np.mean(pool)),
        }


@dataclass
class BiorouterC302Loop:
    """Harness talking to biorouter over HTTP, model id c302.celegans."""

    url: str
    harness: FlyHarness
    stub: bool

    @property
    def n_neurons(self) -> int:
        return int(self.harness.n_neurons)

    @property
    def model_id(self) -> str:
        return str(self.harness.model_id)


def c302_provider(*, real: bool = False) -> Any:
    """Backend listed as ``c302.celegans``. Never Docker. Never downloads."""
    if real:
        backend = running_c302_backend()
        if backend is None:
            raise RuntimeError(
                "no running c302 Model in this environment "
                "(will not start OpenWorm Docker)"
            )
        return backend
    return FakeC302(model_id=C302_MODEL_ID)


def make_registry(*, real: bool = False) -> BioSimRouter:
    """Default fixtures including FakeC302, or a real running Model if asked."""
    backends = create_default_backends()
    backends[C302_MODEL_ID] = c302_provider(real=real)
    return create_registry(backends)


def make_codec(n_neurons: int) -> tuple[TouchEncoder, MotorDecoder]:
    return TouchEncoder(n_neurons), MotorDecoder(n_neurons)


def build_loop(url: str, *, n_neurons: int | None = None) -> BiorouterC302Loop:
    """Wire FlyHarness.step to biorouter model id ``c302.celegans``."""
    client = HttpModelBackend(url, model_id=C302_MODEL_ID, n_neurons=n_neurons)
    n = int(client.n_neurons)
    encoder, decoder = make_codec(n)
    backend = DirectBioSimBackend(url=url, model_id=C302_MODEL_ID, n_neurons=n)
    harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
    return BiorouterC302Loop(
        url=url,
        harness=harness,
        stub=n == C302_N_NEURONS,
    )


def iter_inprocess_loop(*, real: bool = False) -> Iterator[BiorouterC302Loop]:
    """Start a local stdlib biorouter that lists ``c302.celegans``."""
    with running_router(registry=make_registry(real=real)) as (url, _server):
        yield build_loop(url)


def run_scripted_loop(
    loop: BiorouterC302Loop,
    *,
    steps: int = 2,
) -> list[dict[str, Any]]:
    """Tick ``c302.celegans`` through HTTP, then hit an unknown id."""
    traces: list[dict[str, Any]] = []
    loop.harness.reset()
    for i in range(max(1, steps)):
        amount = 0.8 if i else 0.0
        result = loop.harness.step({"touch": amount})
        traces.append(
            {
                "phase": "c302",
                "step": i,
                "model_id": loop.harness.model_id,
                "n_neurons": loop.n_neurons,
                "timestamp": float(result.timestamp),
                "action": dict(result.action),
            }
        )

    unknown = BioSimRouter(remote_url=loop.url)
    forwarded = unknown.select("ghost.sim")
    try:
        forwarded.tick(np.zeros(min(8, loop.n_neurons), dtype=np.float64))
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
    parser.add_argument(
        "--real",
        action="store_true",
        help=(
            "list an already-importable running c302 Model as c302.celegans "
            "(never downloads, never starts OpenWorm Docker)"
        ),
    )
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)
    print(_BANNER)

    if args.real and args.url:
        raise SystemExit(
            " --real is for the in-thread registry. Attach with --url to a "
            "biorouter that already listed c302.celegans (or get HTTP 404)."
        )

    if args.real:
        if not c302_available():
            raise SystemExit(
                " --real needs third-party c302 already importable in this "
                "environment. Will not download connectomes. Will not start "
                "OpenWorm Docker. Deploy a running tick/reset sim and list "
                f"model id {C302_MODEL_ID}, or omit --real to use FakeC302 "
                f"({C302_N_NEURONS} hermaphrodite neurons)."
            )
        if running_c302_backend() is None:
            raise SystemExit(
                " c302 is importable, but it is not a running tick/reset "
                "Model. This repo does not wrap NEURON and will not start "
                "OpenWorm Docker (not a hosted OpenWorm API). Deploy a "
                f"ModelBackend and list {C302_MODEL_ID}, or omit --real."
            )

    def _run(loop: BiorouterC302Loop) -> None:
        print(
            f"url={loop.url} model={loop.model_id} n_neurons={loop.n_neurons} "
            f"hermaphrodite_ref={C302_N_NEURONS} stub={loop.stub}"
        )
        traces = run_scripted_loop(loop, steps=args.steps)
        for row in traces:
            if row["phase"] == "unknown":
                print(
                    f"unknown model: error={row.get('error')} "
                    f"model_id={row.get('model_id')}"
                )
                continue
            print(
                f"{row['phase']} step {row['step']}: model={row['model_id']} "
                f"n_neurons={row['n_neurons']} timestamp={row['timestamp']} "
                f"action={row['action']}"
            )

    if args.url:
        _run(build_loop(args.url))
        return

    with running_router(registry=make_registry(real=args.real)) as (url, _server):
        _run(build_loop(url))


if __name__ == "__main__":
    main()
