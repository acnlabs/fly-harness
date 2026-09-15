"""Tiny sparse demo connectome (tens of neurons, not FlyWire scale)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from fly_harness.brain_state import BrainState

# 24 neurons: 4 sensory, 12 interneurons, 8 motor (2 pools × 4)
NEURON_LABELS: tuple[str, ...] = (
    "sens_touch_left_0",
    "sens_touch_left_1",
    "sens_touch_right_0",
    "sens_touch_right_1",
    *(f"inter_{i}" for i in range(12)),
    "motor_turn_left_0",
    "motor_turn_left_1",
    "motor_forward_0",
    "motor_forward_1",
    "motor_turn_right_0",
    "motor_turn_right_1",
    "motor_brake_0",
    "motor_brake_1",
)

N_NEURONS = len(NEURON_LABELS)
SENSORY_INDICES: tuple[int, ...] = (0, 1, 2, 3)

_SENS_LEFT = (0, 1)
_SENS_RIGHT = (2, 3)
_INTER = tuple(range(4, 16))
_MOTOR_TURN_LEFT = (16, 17)
_MOTOR_FORWARD = (18, 19)
_MOTOR_TURN_RIGHT = (20, 21)
_MOTOR_BRAKE = (22, 23)


def _add_synapses(
    pre: list[int],
    post: list[int],
    weight: list[float],
    pairs: list[tuple[int, int, float]],
) -> None:
    for p, q, w in pairs:
        pre.append(p)
        post.append(q)
        weight.append(w)


def build_reflex_connectome() -> BrainState:
    """Return a 24-neuron sparse connectome for a touch reflex demo.

    Wiring sketch (not biologically validated):

        touch_left  -> interneuron pool -> turn_right motor pool
        touch_right -> interneuron pool -> turn_left motor pool
        weak cross-inhibition between turn pools
        forward pool receives baseline bias via interneurons
    """
    pre: list[int] = []
    post: list[int] = []
    weight: list[float] = []

    # Sensory -> interneurons (sparse fan-out)
    for s in _SENS_LEFT:
        for i in _INTER[:6]:
            _add_synapses(pre, post, weight, [(s, i, 0.9)])
    for s in _SENS_RIGHT:
        for i in _INTER[6:]:
            _add_synapses(pre, post, weight, [(s, i, 0.9)])

    # Interneuron chain (local recurrence)
    for i in range(4, 15):
        _add_synapses(pre, post, weight, [(i, i + 1, 0.4)])

    # Fast monosynaptic reflex arcs (demo-only; polysynaptic paths add later dynamics)
    for s in _SENS_LEFT:
        for m in _MOTOR_TURN_RIGHT:
            _add_synapses(pre, post, weight, [(s, m, 0.85)])
    for s in _SENS_RIGHT:
        for m in _MOTOR_TURN_LEFT:
            _add_synapses(pre, post, weight, [(s, m, 0.85)])

    # Left touch pathway -> right turn (escape away from stimulus)
    for i in _INTER[:6]:
        for m in _MOTOR_TURN_RIGHT:
            _add_synapses(pre, post, weight, [(i, m, 0.7)])
        for m in _MOTOR_BRAKE:
            _add_synapses(pre, post, weight, [(i, m, 0.3)])

    # Right touch pathway -> left turn
    for i in _INTER[6:]:
        for m in _MOTOR_TURN_LEFT:
            _add_synapses(pre, post, weight, [(i, m, 0.7)])
        for m in _MOTOR_BRAKE:
            _add_synapses(pre, post, weight, [(i, m, 0.3)])

    # Baseline forward bias through interneurons
    for i in _INTER[2:10]:
        for m in _MOTOR_FORWARD:
            _add_synapses(pre, post, weight, [(i, m, 0.25)])

    weights = sparse.coo_matrix(
        (weight, (pre, post)),
        shape=(N_NEURONS, N_NEURONS),
    ).tocsr()

    return BrainState(
        potentials=np.zeros(N_NEURONS, dtype=np.float64),
        weights=weights,
        timestamp=0.0,
        neuron_labels=NEURON_LABELS,
    )
