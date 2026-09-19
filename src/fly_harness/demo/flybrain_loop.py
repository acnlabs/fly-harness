"""Example: FlyHarness.step with a third-party flybrain Model.

Wiring flybrain is using the harness, not a harness feature. Default run uses
FakeFlyBrain (32 neurons, no download). Pass ``--real`` only when ``flybrain``
is installed **and** MaleCNS files are already in ~/fly-data; this script will
not download them. biorouter lists ``flybrain.malecns`` when extra + data exist.

    python examples/flybrain_loop.py
    python examples/flybrain_loop.py --real
    python -m fly_harness.demo.flybrain_loop
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from fly_harness.flybrain import (
    MALECNS_N_NEURONS,
    FakeFlyBrain,
    FlyBrainBackend,
    FlyBrainInjectEncoder,
    FlyBrainReadoutDecoder,
    flybrain_available,
    flybrain_data_available,
)
from fly_harness.harness import FlyHarness

_BANNER = (
    "Example: FlyHarness.step with a third-party flybrain Model. "
    "obs → encode → Model.tick → decode → action. flybrain is a third-party LIF over MaleCNS v1.0 "
    "(166,700 neurons, not 160k parameters). Wiring it is using the harness, "
    "not a harness feature. Not Google-hosted, not the FlyWire website, not a "
    "consciousness or upload claim. Default loop is a FakeFlyBrain stub "
    "(no ~/fly-data download). biorouter lists flybrain.malecns when extra + data exist."
)


def build_loop(*, real: bool = False, download: bool = False) -> tuple[FlyHarness, FlyBrainBackend]:
    if real:
        backend = FlyBrainBackend.from_installed(download=download)
        n = backend.n_neurons
        cells = getattr(backend.brain, "cells", None)
        inject_idx = np.arange(min(4, n), dtype=np.int64)
        readout_idx = np.arange(max(0, n - 4), n, dtype=np.int64)
        if callable(cells):
            loom = np.asarray(cells(["LC4", "LPLC2"]), dtype=np.int64)
            dn = np.asarray(cells(["descending_neuron"]), dtype=np.int64)
            if loom.size:
                inject_idx = loom[: min(32, loom.size)]
            if dn.size:
                readout_idx = dn[: min(64, dn.size)]
        encoder = FlyBrainInjectEncoder(n, channels={"loom": inject_idx})
        decoder = FlyBrainReadoutDecoder(n, readout_indices=readout_idx)
    else:
        brain = FakeFlyBrain()
        backend = FlyBrainBackend(brain, model_id="flybrain.fake")
        encoder = FlyBrainInjectEncoder(
            brain.n, channels={"loom": tuple(int(i) for i in brain.cells(["sens"]))}
        )
        decoder = FlyBrainReadoutDecoder(
            brain.n, readout_indices=brain.cells(["descending_neuron"])
        )
    harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
    return harness, backend


def run_scripted_loop(harness: FlyHarness, steps: int = 3) -> list[dict[str, Any]]:
    harness.reset()
    traces: list[dict[str, Any]] = []
    for i in range(max(1, steps)):
        amount = 0.8 if i else 0.0
        result = harness.step({"loom": amount})
        traces.append(
            {
                "step": i,
                "model_id": harness.model_id,
                "n_neurons": harness.n_neurons,
                "timestamp": float(result.timestamp),
                "action": dict(result.action),
            }
        )
    return traces


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=_BANNER)
    parser.add_argument(
        "--real",
        action="store_true",
        help="use installed flybrain + on-disk MaleCNS files (never downloads)",
    )
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    print(_BANNER)
    if args.real:
        if not flybrain_available() or not flybrain_data_available():
            raise SystemExit(
                " --real needs fly-harness[flybrain] and MaleCNS files already on disk "
                f"(expected {MALECNS_N_NEURONS} neurons). Will not download. "
                "Install extra and run `flybrain download` yourself, or omit --real."
            )
    harness, backend = build_loop(real=args.real, download=False)
    print(
        f"model={harness.model_id} n_neurons={harness.n_neurons} "
        f"malecns_ref={MALECNS_N_NEURONS} stub={isinstance(backend.brain, FakeFlyBrain)}"
    )
    if harness.n_neurons == MALECNS_N_NEURONS:
        print("using real MaleCNS-sized third-party Model (dump on disk, not a hosted Google/FlyWire service)")
    traces = run_scripted_loop(harness, steps=args.steps)
    for row in traces:
        print(
            f"step {row['step']}: n_neurons={row['n_neurons']} "
            f"timestamp={row['timestamp']} action={row['action']}"
        )


if __name__ == "__main__":
    main()
