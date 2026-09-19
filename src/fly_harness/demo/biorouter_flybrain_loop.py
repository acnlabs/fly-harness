"""Example: FlyHarness.step via local biorouter to model id flybrain.malecns.

Wiring flybrain is using the harness, not a harness feature. OpenRouter lists
a provider; biorouter lists ``flybrain.malecns`` the same way.

Default in-thread run registers a fake flybrain-shaped backend (32 neurons,
no download). Pass ``--real`` only when ``flybrain`` is installed **and**
MaleCNS files are already on disk. ``--url`` attaches to a process you
already started; missing ``flybrain.malecns`` is HTTP 404.

    python examples/biorouter_flybrain_loop.py
    python examples/biorouter_flybrain_loop.py --real
    python -m fly_harness.demo.biorouter_flybrain_loop
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from fly_harness.backend import BioSimRouter, DirectBioSimBackend, UnknownModelError
from fly_harness.flybrain import (
    FAKE_N_NEURONS,
    MALECNS_N_NEURONS,
    FakeFlyBrain,
    FlyBrainBackend,
    FlyBrainInjectEncoder,
    FlyBrainReadoutDecoder,
    flybrain_available,
    flybrain_data_available,
)
from fly_harness.harness import FlyHarness
from fly_harness.http_backend import HttpModelBackend
from fly_harness.router import FLYBRAIN_MODEL_ID, create_default_backends, create_registry
from fly_harness.router.server import running_router

_BANNER = (
    "biorouter → flybrain.malecns example: obs → encode → Model.tick → decode → action. "
    "OpenRouter lists a provider id; biorouter lists flybrain.malecns. "
    "FlyHarness.step goes through local stdlib HTTP (not a harness feature). "
    "Default is FakeFlyBrain (32 neurons, not 160k parameters, not 166,700). "
    "Not Google-hosted, not the FlyWire website, not a consciousness or upload "
    "claim, not a marketplace. Never downloads MaleCNS."
)


@dataclass
class BiorouterFlybrainLoop:
    """Harness talking to biorouter over HTTP, model id flybrain.malecns."""

    url: str
    harness: FlyHarness
    stub: bool

    @property
    def n_neurons(self) -> int:
        return int(self.harness.n_neurons)

    @property
    def model_id(self) -> str:
        return str(self.harness.model_id)


def flybrain_provider(*, real: bool = False) -> FlyBrainBackend:
    """Backend listed as ``flybrain.malecns``. Never downloads MaleCNS."""
    if real:
        return FlyBrainBackend.from_installed(
            download=False,
            model_id=FLYBRAIN_MODEL_ID,
        )
    brain = FakeFlyBrain()
    return FlyBrainBackend(brain, model_id=FLYBRAIN_MODEL_ID)


def make_registry(*, real: bool = False) -> BioSimRouter:
    """Default fixtures plus ``flybrain.malecns`` (fake unless ``real=True``)."""
    backends = create_default_backends()
    backends[FLYBRAIN_MODEL_ID] = flybrain_provider(real=real)
    return create_registry(backends)


def make_codec(n_neurons: int) -> tuple[FlyBrainInjectEncoder, FlyBrainReadoutDecoder]:
    if n_neurons == FAKE_N_NEURONS:
        layout = FakeFlyBrain()
        encoder = FlyBrainInjectEncoder(
            n_neurons,
            channels={"loom": tuple(int(i) for i in layout.cells(["sens"]))},
        )
        decoder = FlyBrainReadoutDecoder(
            n_neurons,
            readout_indices=layout.cells(["descending_neuron"]),
        )
        return encoder, decoder
    encoder = FlyBrainInjectEncoder(
        n_neurons,
        channels={"loom": tuple(range(min(4, n_neurons)))},
    )
    decoder = FlyBrainReadoutDecoder(n_neurons)
    return encoder, decoder


def build_loop(url: str, *, n_neurons: int | None = None) -> BiorouterFlybrainLoop:
    """Wire FlyHarness.step to biorouter model id ``flybrain.malecns``."""
    client = HttpModelBackend(url, model_id=FLYBRAIN_MODEL_ID, n_neurons=n_neurons)
    n = int(client.n_neurons)
    encoder, decoder = make_codec(n)
    backend = DirectBioSimBackend(url=url, model_id=FLYBRAIN_MODEL_ID, n_neurons=n)
    harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
    return BiorouterFlybrainLoop(
        url=url,
        harness=harness,
        stub=n == FAKE_N_NEURONS and n != MALECNS_N_NEURONS,
    )


def iter_inprocess_loop(*, real: bool = False) -> Iterator[BiorouterFlybrainLoop]:
    """Start a local stdlib biorouter that lists ``flybrain.malecns``."""
    with running_router(registry=make_registry(real=real)) as (url, _server):
        yield build_loop(url)


def run_scripted_loop(
    loop: BiorouterFlybrainLoop,
    *,
    steps: int = 2,
) -> list[dict[str, Any]]:
    """Tick ``flybrain.malecns`` through HTTP, then hit an unknown id."""
    traces: list[dict[str, Any]] = []
    loop.harness.reset()
    for i in range(max(1, steps)):
        amount = 0.8 if i else 0.0
        result = loop.harness.step({"loom": amount})
        traces.append(
            {
                "phase": "flybrain",
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
        help="list installed flybrain + on-disk MaleCNS as flybrain.malecns (never downloads)",
    )
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)
    print(_BANNER)

    if args.real and args.url:
        raise SystemExit(
            " --real is for the in-thread registry. Attach with --url to a "
            "biorouter that already listed flybrain.malecns (or get HTTP 404)."
        )

    if args.real:
        if not flybrain_available() or not flybrain_data_available():
            raise SystemExit(
                " --real needs fly-harness[flybrain] and MaleCNS files already on disk "
                f"(expected {MALECNS_N_NEURONS} neurons). Will not download. "
                "Install extra and run `flybrain download` yourself, or omit --real."
            )

    def _run(loop: BiorouterFlybrainLoop) -> None:
        print(
            f"url={loop.url} model={loop.model_id} n_neurons={loop.n_neurons} "
            f"malecns_ref={MALECNS_N_NEURONS} stub={loop.stub}"
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
