"""Same FlyGym body, same loop, swap the Model.

Product difference: keep ``obs → FlyHarness.step → action`` and the body;
change only the docked ``ModelBackend``. Backend A is the default toy
in-process LIF. Backend B is biorouter id ``flybrain.malecns``.

Reuses the existing FlyGym body-loop + biorouter examples. Does not fork
the kernel. Does not claim better walking than FlyGym's own controllers.
Missing flybrain extra or on-disk MaleCNS → skip B (never a fake 166,700
neuron brain, never a download).

    python examples/swap_brain.py
    python examples/swap_brain.py --real
    python examples/swap_brain.py --try-flygym
    python -m fly_harness.demo.swap_brain
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from fly_harness.backend import DEFAULT_LIF_MODEL_ID, DirectBioSimBackend, InProcessLifBackend
from fly_harness.demo.body_loop import (
    StubFlyGymEnv,
    contact_observation,
    resolve_body_env,
)
from fly_harness.demo.biorouter_flybrain_loop import make_registry
from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome
from fly_harness.flybrain import FAKE_N_NEURONS, MALECNS_N_NEURONS, flybrain_available, flybrain_data_available
from fly_harness.flygym import FlyGymDecoder, FlyGymEncoder, FlyGymHarnessEnv, coerce_action
from fly_harness.flygym.detect import flygym_available
from fly_harness.harness import FlyHarness
from fly_harness.http_backend import HttpModelBackend
from fly_harness.router import FLYBRAIN_MODEL_ID
from fly_harness.router.server import running_router

_BANNER = (
    "same loop, swap Model: same FlyGym body + obs → FlyHarness.step → action; "
    "only the Model changes. A = toy in-process LIF "
    f"({DEFAULT_LIF_MODEL_ID}). B = biorouter id {FLYBRAIN_MODEL_ID}. "
    "Usage of extras, not a harness feature. Core does not bind a vendor Model. "
    "FlyGym / CPG are not the harness kernel; FlyHarness.step is brain-level "
    "above the motor CPG. "
    "This does not claim better walking than FlyGym's own controllers. "
    "Default B is FakeFlyBrain listed under that id (32 neurons, not 160k "
    "parameters, not 166,700). --real needs fly-harness[flybrain] and MaleCNS "
    "already on disk; otherwise skip B (will not download, will not pretend). "
    "Not Google-hosted, not the FlyWire website, not a consciousness or upload "
    "claim, not a marketplace."
)

_OBS_SCHEDULE = (
    {"touch_left": 0.0, "touch_right": 0.0},
    {"touch_left": 1.0, "touch_right": 0.0},
    {"touch_left": 0.0, "touch_right": 1.0},
)


@dataclass
class BackendRun:
    """One pass of the shared loop against one Model."""

    model_id: str
    n_neurons: int
    stub_model: bool
    traces: list[dict[str, Any]]
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class SwapRun:
    """Same body env, same loop function, two Models (or A + honest skip of B)."""

    body_name: str
    stub_body: bool
    backend_a: BackendRun
    backend_b: BackendRun
    same_body: bool


def harness_for_backend(backend: Any) -> FlyHarness:
    """FlyGym encoder/decoder sized to this Model. Loop code stays the same."""
    n = int(backend.n_neurons)
    return FlyHarness(
        encoder=FlyGymEncoder(n),
        decoder=FlyGymDecoder(n),
        backend=backend,
    )


def toy_lif_backend() -> InProcessLifBackend:
    """Backend A: default toy in-process LIF (24-neuron fixture, not biology)."""
    state = build_reflex_connectome()
    return InProcessLifBackend(state, sensory_indices=SENSORY_INDICES)


def attach_harness(body: FlyGymHarnessEnv, harness: FlyHarness) -> FlyGymHarnessEnv:
    """Swap the Model on an existing body wrapper. Same env, same ``body.step``."""
    body.harness = harness
    decoder_pose = getattr(harness.decoder, "rest_pose", None)
    if decoder_pose is not None:
        body.rest_pose = np.asarray(decoder_pose, dtype=np.float64).reshape(-1)
    return body


def run_identical_loop(body: FlyGymHarnessEnv, *, steps: int = 3) -> list[dict[str, Any]]:
    """The loop: obs → FlyHarness.step → FlyGym action. Shared by A and B."""
    body.reset()
    traces: list[dict[str, Any]] = []
    for i in range(max(1, steps)):
        obs = contact_observation(**_OBS_SCHEDULE[i % len(_OBS_SCHEDULE)])
        result = body.step(obs)
        action = coerce_action(result.action)
        traces.append(
            {
                "step": i,
                "model_id": str(body.harness.model_id),
                "n_neurons": int(body.harness.n_neurons),
                "timestamp": float(result.harness.timestamp),
                "joints": action.joints.tolist(),
                "adhesion": None if action.adhesion is None else action.adhesion.tolist(),
            }
        )
    return traces


def _skip_b(reason: str) -> BackendRun:
    return BackendRun(
        model_id=FLYBRAIN_MODEL_ID,
        n_neurons=0,
        stub_model=True,
        traces=[],
        skipped=True,
        skip_reason=reason,
    )


def real_flybrain_ready() -> bool:
    return bool(flybrain_available() and flybrain_data_available())


def skip_real_b_reason() -> str:
    return (
        f"skip B ({FLYBRAIN_MODEL_ID}): fly-harness[flybrain] extra or MaleCNS "
        f"files missing (expected {MALECNS_N_NEURONS} neurons). Will not download. "
        f"Will not pretend a {MALECNS_N_NEURONS}-neuron brain is running."
    )


def run_backend_a(body: FlyGymHarnessEnv, *, steps: int) -> BackendRun:
    attach_harness(body, harness_for_backend(toy_lif_backend()))
    traces = run_identical_loop(body, steps=steps)
    n = int(body.harness.n_neurons)
    return BackendRun(
        model_id=str(body.harness.model_id),
        n_neurons=n,
        stub_model=True,
        traces=traces,
    )


def run_backend_b(
    body: FlyGymHarnessEnv,
    *,
    steps: int,
    url: str | None,
    real: bool,
) -> BackendRun:
    """Biorouter id flybrain.malecns. Skip honestly if the Model cannot tick."""
    if real and url:
        return _skip_b(
            "skip B: --real is for the in-thread registry. Attach with --url to a "
            f"biorouter that already listed {FLYBRAIN_MODEL_ID} (or get HTTP 404)."
        )
    if real and not real_flybrain_ready():
        return _skip_b(skip_real_b_reason())

    def _tick_listed(router_url: str) -> BackendRun:
        try:
            client = HttpModelBackend(router_url, model_id=FLYBRAIN_MODEL_ID)
            n = int(client.n_neurons)
        except Exception as exc:
            return _skip_b(
                f"skip B ({FLYBRAIN_MODEL_ID}): {type(exc).__name__}: {exc}. "
                f"Will not pretend a {MALECNS_N_NEURONS}-neuron brain is running."
            )
        backend = DirectBioSimBackend(url=router_url, model_id=FLYBRAIN_MODEL_ID, n_neurons=n)
        attach_harness(body, harness_for_backend(backend))
        traces = run_identical_loop(body, steps=steps)
        return BackendRun(
            model_id=str(body.harness.model_id),
            n_neurons=int(body.harness.n_neurons),
            stub_model=n == FAKE_N_NEURONS,
            traces=traces,
        )

    if url:
        return _tick_listed(url)

    with running_router(registry=make_registry(real=real)) as (router_url, _server):
        return _tick_listed(router_url)


def run_swap(
    *,
    steps: int = 3,
    real: bool = False,
    try_real_flygym: bool = False,
    url: str | None = None,
    env: Any | None = None,
) -> SwapRun:
    body_env = resolve_body_env(env, try_real_flygym=try_real_flygym)
    placeholder = harness_for_backend(toy_lif_backend())
    body = FlyGymHarnessEnv(body_env, placeholder)
    backend_a = run_backend_a(body, steps=steps)
    env_id_before = id(body.env)
    backend_b = run_backend_b(body, steps=steps, url=url, real=real)
    return SwapRun(
        body_name=type(body.env).__name__,
        stub_body=isinstance(body.env, StubFlyGymEnv),
        backend_a=backend_a,
        backend_b=backend_b,
        same_body=id(body.env) == env_id_before,
    )


def _print_backend(label: str, run: BackendRun) -> None:
    if run.skipped:
        print(f"{label}: SKIPPED {run.skip_reason}")
        return
    print(
        f"{label}: model_id={run.model_id} n_neurons={run.n_neurons} "
        f"stub_model={run.stub_model} malecns_ref={MALECNS_N_NEURONS}"
    )
    for row in run.traces:
        joints = np.asarray(row["joints"])
        print(
            f"  loop step {row['step']}: model={row['model_id']} "
            f"n_neurons={row['n_neurons']} timestamp={row['timestamp']} "
            f"body_joints[:3]={joints[:3]}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=_BANNER)
    parser.add_argument(
        "--url",
        default=None,
        help="attach B to an already-running biorouter (default: in-thread stdlib server)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="B = installed flybrain + on-disk MaleCNS (never downloads); skip B if missing",
    )
    parser.add_argument(
        "--try-flygym",
        action="store_true",
        help="try a real NeuroMechFly sim; fall back to the in-process stub",
    )
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    print(_BANNER)

    result = run_swap(
        steps=args.steps,
        real=args.real,
        try_real_flygym=args.try_flygym,
        url=args.url,
    )
    print(
        f"body={result.body_name} stub_body={result.stub_body} "
        f"same_body={result.same_body} flygym={flygym_available()} "
        "loop=obs→FlyHarness.step→action"
    )
    _print_backend("A", result.backend_a)
    _print_backend("B", result.backend_b)
    if result.backend_b.skipped:
        print("B skipped honestly; A still ran the same loop.")
    else:
        print(
            "same loop, two model_ids: "
            f"{result.backend_a.model_id} → {result.backend_b.model_id}"
        )


if __name__ == "__main__":
    main()
