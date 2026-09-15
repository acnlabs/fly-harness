"""Reflex loop demo: touch stimulus -> sparse connectome -> motor action."""

from __future__ import annotations

from fly_harness.demo.codec import ReflexAction, ReflexDecoder, ReflexEncoder, TouchObservation
from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome
from fly_harness.harness import FlyHarness


def run_reflex_loop(steps: int = 6) -> list[ReflexAction]:
    """Run a short scripted reflex loop and return emitted actions."""
    state = build_reflex_connectome()
    harness = FlyHarness(
        state=state,
        encoder=ReflexEncoder(),
        decoder=ReflexDecoder(),
        decay=0.55,
        gain=0.45,
        sensory_indices=SENSORY_INDICES,
    )

    schedule: list[TouchObservation] = [
        TouchObservation(),
        TouchObservation(touch_left=1.0),
        TouchObservation(touch_left=1.0),
        TouchObservation(),
        TouchObservation(touch_right=1.0),
        TouchObservation(touch_right=1.0),
    ]

    actions: list[ReflexAction] = []
    for obs in schedule[:steps]:
        result = harness.step(obs)
        actions.append(result.action)
    return actions


def main() -> None:
    state = build_reflex_connectome()
    harness = FlyHarness(
        state=state,
        encoder=ReflexEncoder(),
        decoder=ReflexDecoder(),
        decay=0.55,
        gain=0.45,
        sensory_indices=SENSORY_INDICES,
    )

    print("fly-harness v0.2 — reflex loop demo")
    print(f"connectome: {state.n_neurons} neurons, {state.weights.nnz} synapses (sparse)")
    print()

    schedule = [
        ("idle", TouchObservation()),
        ("touch left", TouchObservation(touch_left=1.0)),
        ("touch left (hold)", TouchObservation(touch_left=1.0)),
        ("release", TouchObservation()),
        ("touch right", TouchObservation(touch_right=1.0)),
        ("touch right (hold)", TouchObservation(touch_right=1.0)),
    ]

    for label, obs in schedule:
        result = harness.step(obs)
        action = result.action
        print(
            f"[t={result.timestamp:4.1f}] {label:20s} "
            f"-> turn={action.turn:+d} forward={action.forward:.2f} brake={action.brake:.2f}"
        )

    print()
    print("Done. This is a toy sparse microkernel, not a FlyWire upload.")


if __name__ == "__main__":
    main()
