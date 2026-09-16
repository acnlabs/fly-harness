"""In-process ModelBackend registry for the optional local HTTP router.

Default ids: the 24-neuron in-process LIF fixture plus FakeDeployedSim
stand-ins. This is a local map, not a marketplace catalog.
"""

from __future__ import annotations

from typing import Mapping

from fly_harness.backend import (
    DEFAULT_LIF_MODEL_ID,
    BioSimRouter,
    FakeDeployedSim,
    InProcessLifBackend,
)
from fly_harness.protocols import ModelBackend

DEFAULT_FAKE_MODEL_ID = "fake.deployed"
DEFAULT_FAKE_GAIN_MODEL_ID = "fake.deployed.gain"
DEFAULT_FAKE_N_NEURONS = 8


def create_default_backends() -> dict[str, ModelBackend]:
    """LIF reflex fixture + FakeDeployedSim ids for local docking tests."""
    from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome

    lif = InProcessLifBackend(
        build_reflex_connectome(),
        sensory_indices=SENSORY_INDICES,
        model_id=DEFAULT_LIF_MODEL_ID,
    )
    fake = FakeDeployedSim(
        DEFAULT_FAKE_N_NEURONS,
        model_id=DEFAULT_FAKE_MODEL_ID,
        scale=1.0,
    )
    fake_gain = FakeDeployedSim(
        DEFAULT_FAKE_N_NEURONS,
        model_id=DEFAULT_FAKE_GAIN_MODEL_ID,
        scale=3.0,
    )
    return {
        lif.model_id: lif,
        fake.model_id: fake,
        fake_gain.model_id: fake_gain,
    }


def create_registry(
    backends: Mapping[str, ModelBackend] | None = None,
) -> BioSimRouter:
    """In-process router used by the HTTP process. Unknown ids error."""
    return BioSimRouter(create_default_backends() if backends is None else backends)
