"""Example: FlyGym obs → FlyHarness.step → local biorouter → listed model → body.

Wiring FlyGym / flybrain is using the harness, not a harness feature. Default
CI path: in-process body stub + FakeFlyBrain listed as ``flybrain.malecns``
(32 neurons, no MaleCNS download, no MuJoCo).

    python examples/body_loop.py
    python examples/body_loop.py --real
    python examples/body_loop.py --try-flygym
    python -m fly_harness.demo.body_loop
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from fly_harness.backend import BioSimRouter, DirectBioSimBackend, UnknownModelError
from fly_harness.demo.biorouter_flybrain_loop import make_registry
from fly_harness.flybrain import FAKE_N_NEURONS, MALECNS_N_NEURONS, flybrain_available, flybrain_data_available
from fly_harness.flygym import (
    N_LEG_JOINTS,
    FlyGymDecoder,
    FlyGymEncoder,
    FlyGymHarnessEnv,
    coerce_action,
)
from fly_harness.flygym.detect import flygym_available
from fly_harness.harness import FlyHarness
from fly_harness.http_backend import HttpModelBackend
from fly_harness.router import FLYBRAIN_MODEL_ID
from fly_harness.router.server import running_router

_BANNER = (
    "body-loop example: FlyGym obs → FlyHarness.step → local biorouter → "
    "flybrain.malecns → decode FlyGym action. obs → encode → Model.tick → decode → action. "
    "Usage of extras, not a harness feature. Default is StubFlyGymEnv + "
    "FakeFlyBrain (32 neurons, not 160k parameters, not 166,700). "
    "Not Google-hosted, not the FlyWire website, not a consciousness or upload "
    "claim, not a marketplace. Never downloads MaleCNS."
)


class StubFlyGymEnv:
    """In-process NeuroMechFly stand-in so the example runs without MuJoCo."""

    def __init__(self) -> None:
        self.actions: list[dict[str, np.ndarray]] = []
        self.obs: dict[str, np.ndarray] = {
            "joints": np.zeros((3, N_LEG_JOINTS), dtype=np.float64),
            "contact_forces": np.zeros((36, 3), dtype=np.float64),
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return self.obs, {}

    def step(self, action: dict[str, np.ndarray]) -> tuple:
        self.actions.append(action)
        return self.obs, 0.0, False, False, {}


@dataclass
class BodyLoop:
    """FlyGym-shaped body driven through biorouter model id flybrain.malecns."""

    url: str
    harness: FlyHarness
    body: FlyGymHarnessEnv
    stub_body: bool
    stub_model: bool

    @property
    def n_neurons(self) -> int:
        return int(self.harness.n_neurons)

    @property
    def model_id(self) -> str:
        return str(self.harness.model_id)


def resolve_body_env(env: Any | None, *, try_real_flygym: bool) -> Any:
    if env is not None:
        return env
    if try_real_flygym:
        try:
            from fly_harness.flygym.wrapper import try_make_neuromechfly_sim

            return try_make_neuromechfly_sim()
        except Exception:
            return StubFlyGymEnv()
    return StubFlyGymEnv()


def contact_observation(*, touch_left: float = 0.0, touch_right: float = 0.0) -> dict[str, np.ndarray]:
    contact = np.zeros((36, 3), dtype=np.float64)
    contact[:18, 2] = float(touch_left)
    contact[18:, 2] = float(touch_right)
    return {
        "joints": np.zeros((3, N_LEG_JOINTS), dtype=np.float64),
        "contact_forces": contact,
    }


def build_loop(
    url: str,
    *,
    env: Any | None = None,
    try_real_flygym: bool = False,
    n_neurons: int | None = None,
) -> BodyLoop:
    """Wire FlyGym obs through FlyHarness.step to biorouter ``flybrain.malecns``."""
    client = HttpModelBackend(url, model_id=FLYBRAIN_MODEL_ID, n_neurons=n_neurons)
    n = int(client.n_neurons)
    encoder = FlyGymEncoder(n)
    decoder = FlyGymDecoder(n)
    backend = DirectBioSimBackend(url=url, model_id=FLYBRAIN_MODEL_ID, n_neurons=n)
    harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
    body_env = resolve_body_env(env, try_real_flygym=try_real_flygym)
    body = FlyGymHarnessEnv(body_env, harness)
    body.reset()
    return BodyLoop(
        url=url,
        harness=harness,
        body=body,
        stub_body=isinstance(body_env, StubFlyGymEnv),
        stub_model=n == FAKE_N_NEURONS,
    )


def iter_inprocess_loop(
    *,
    real: bool = False,
    try_real_flygym: bool = False,
    env: Any | None = None,
) -> Iterator[BodyLoop]:
    """Start a local stdlib biorouter listing ``flybrain.malecns`` and yield a loop."""
    with running_router(registry=make_registry(real=real)) as (url, _server):
        yield build_loop(url, env=env, try_real_flygym=try_real_flygym)


def run_scripted_loop(loop: BodyLoop, *, steps: int = 3) -> list[dict[str, Any]]:
    """Body obs → harness.step via HTTP → joints on the env; then unknown id 404."""
    loop.body.reset()
    schedule = (
        {"touch_left": 0.0, "touch_right": 0.0},
        {"touch_left": 1.0, "touch_right": 0.0},
        {"touch_left": 0.0, "touch_right": 1.0},
    )
    traces: list[dict[str, Any]] = []
    for i in range(max(1, steps)):
        obs = contact_observation(**schedule[i % len(schedule)])
        result = loop.body.step(obs)
        action = coerce_action(result.action)
        traces.append(
            {
                "phase": "body",
                "step": i,
                "model_id": loop.model_id,
                "n_neurons": loop.n_neurons,
                "timestamp": float(result.harness.timestamp),
                "joints": action.joints.tolist(),
                "adhesion": None if action.adhesion is None else action.adhesion.tolist(),
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
    parser.add_argument(
        "--try-flygym",
        action="store_true",
        help="try a real NeuroMechFly sim; fall back to the in-process stub",
    )
    parser.add_argument("--steps", type=int, default=3)
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

    def _run(loop: BodyLoop) -> None:
        print(
            f"url={loop.url} model={loop.model_id} n_neurons={loop.n_neurons} "
            f"malecns_ref={MALECNS_N_NEURONS} stub_model={loop.stub_model} "
            f"body={type(loop.body.env).__name__} stub_body={loop.stub_body} "
            f"flygym={flygym_available()}"
        )
        traces = run_scripted_loop(loop, steps=args.steps)
        for row in traces:
            if row["phase"] == "unknown":
                print(
                    f"unknown model: error={row.get('error')} "
                    f"model_id={row.get('model_id')}"
                )
                continue
            joints = np.asarray(row["joints"])
            print(
                f"{row['phase']} step {row['step']}: model={row['model_id']} "
                f"n_neurons={row['n_neurons']} timestamp={row['timestamp']} "
                f"body_joints[:3]={joints[:3]}"
            )

    if args.url:
        _run(build_loop(args.url, try_real_flygym=args.try_flygym))
        return

    with running_router(registry=make_registry(real=args.real)) as (url, _server):
        _run(build_loop(url, try_real_flygym=args.try_flygym))


if __name__ == "__main__":
    main()
