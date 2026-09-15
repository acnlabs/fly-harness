"""Tests for FlyHarness stepping and reflex behavior."""

from fly_harness.demo.codec import ReflexDecoder, ReflexEncoder, TouchObservation
from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome
from fly_harness.demo.reflex import run_reflex_loop
from fly_harness.harness import FlyHarness


def _make_harness(**kwargs: object) -> FlyHarness:
    defaults = {
        "state": build_reflex_connectome(),
        "encoder": ReflexEncoder(),
        "decoder": ReflexDecoder(),
        "sensory_indices": SENSORY_INDICES,
    }
    defaults.update(kwargs)
    return FlyHarness(**defaults)


def test_step_returns_action_and_advances_timestamp() -> None:
    harness = _make_harness()
    result = harness.step(TouchObservation())
    assert result.timestamp == 1.0
    assert hasattr(result.action, "turn")
    assert result.potentials.shape == (24,)


def test_left_touch_biases_right_turn() -> None:
    harness = _make_harness(decay=0.2, gain=0.45)
    harness.step(TouchObservation())
    harness.step(TouchObservation(touch_left=1.0))
    result = harness.step(TouchObservation(touch_left=1.0))
    assert result.action.turn == 1


def test_right_touch_biases_left_turn() -> None:
    harness = _make_harness(decay=0.2, gain=0.45)
    harness.step(TouchObservation())
    harness.step(TouchObservation(touch_right=1.0))
    result = harness.step(TouchObservation(touch_right=1.0))
    assert result.action.turn == -1


def test_reflex_loop_runs_expected_steps() -> None:
    actions = run_reflex_loop(steps=6)
    assert len(actions) == 6
    assert actions[2].turn == 1
    assert actions[5].turn == -1
