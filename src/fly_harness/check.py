"""Load-check: prove the harness loaded and can take one step.

Default path is the 24-neuron toy in-process LIF fixture. No extras.
Not a chat shell, web UI, or model catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from fly_harness.backend import DEFAULT_LIF_MODEL_ID
from fly_harness.demo.codec import ReflexDecoder, ReflexEncoder, TouchObservation
from fly_harness.demo.connectome import N_NEURONS, SENSORY_INDICES, build_reflex_connectome
from fly_harness.harness import FlyHarness

DEFAULT_OBS = TouchObservation(touch_left=1.0, touch_right=0.0)


@dataclass(frozen=True)
class LoadCheckReport:
    """One fixture step: enough to show encode → tick → decode is alive."""

    model_id: str
    n_neurons: int
    timestamp: float
    obs: dict[str, float]
    action: dict[str, Any]

    def format_text(self) -> str:
        obs = " ".join(f"{key}={value}" for key, value in self.obs.items())
        action = " ".join(f"{key}={_format_action_value(value)}" for key, value in self.action.items())
        return (
            f"model_id: {self.model_id}\n"
            f"n_neurons: {self.n_neurons}\n"
            f"timestamp: {self.timestamp}\n"
            f"obs: {obs}\n"
            f"action: {action}\n"
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _format_action_value(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    return f"{value:.4f}"


def build_fixture_harness() -> FlyHarness:
    """Default toy LIF: 24-neuron reflex fixture, no extras."""
    return FlyHarness(
        build_reflex_connectome(),
        ReflexEncoder(),
        ReflexDecoder(),
        sensory_indices=SENSORY_INDICES,
        decay=0.2,
        gain=0.45,
    )


def run_load_check(
    observation: TouchObservation | Mapping[str, float] | None = None,
) -> LoadCheckReport:
    """Run one `FlyHarness.step` on the default in-process LIF fixture."""
    obs = DEFAULT_OBS if observation is None else observation
    if not isinstance(obs, TouchObservation):
        obs = TouchObservation(
            touch_left=float(obs.get("touch_left", 0.0)),
            touch_right=float(obs.get("touch_right", 0.0)),
        )
    harness = build_fixture_harness()
    result = harness.step(obs)
    action = result.action
    return LoadCheckReport(
        model_id=str(harness.model_id),
        n_neurons=int(harness.n_neurons),
        timestamp=float(result.timestamp),
        obs={"touch_left": float(obs.touch_left), "touch_right": float(obs.touch_right)},
        action={
            "turn": int(action.turn),
            "forward": float(action.forward),
            "brake": float(action.brake),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fly-harness-check",
        description=(
            "Prove fly-harness loaded: print model_id, n_neurons, timestamp, "
            "and one obs→action step on the toy in-process LIF fixture. "
            "No extras. Not a chat shell, web UI, or catalog."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the same fields as JSON",
    )
    args = parser.parse_args(argv)
    report = run_load_check()
    if report.model_id != DEFAULT_LIF_MODEL_ID or report.n_neurons != N_NEURONS:
        print(
            "load-check failed: expected the in-process LIF fixture "
            f"({DEFAULT_LIF_MODEL_ID}, {N_NEURONS} neurons)",
            file=sys.stderr,
        )
        return 1
    sys.stdout.write(report.to_json() if args.json else report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
