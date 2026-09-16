"""In-process ModelBackend registry for the optional biorouter process.

Default ids: the 24-neuron in-process LIF fixture, FakeDeployedSim
stand-ins, and FakeC302 listed as ``c302.celegans`` (302 hermaphrodite
neurons, no OpenWorm). If the third-party flybrain extra and on-disk
MaleCNS files are present, ``flybrain.malecns`` is listed like an
OpenRouter provider id. Missing flybrain → omit (HTTP 404). Never
downloads MaleCNS. Never starts OpenWorm Docker. Not a marketplace.
"""

from __future__ import annotations

from typing import Mapping

from fly_harness.backend import (
    DEFAULT_LIF_MODEL_ID,
    BioSimRouter,
    FakeDeployedSim,
    InProcessLifBackend,
)
from fly_harness.c302 import C302_MODEL_ID, FakeC302
from fly_harness.protocols import ModelBackend

DEFAULT_FAKE_MODEL_ID = "fake.deployed"
DEFAULT_FAKE_GAIN_MODEL_ID = "fake.deployed.gain"
DEFAULT_FAKE_N_NEURONS = 8
FLYBRAIN_MODEL_ID = "flybrain.malecns"


def try_register_flybrain(backends: dict[str, ModelBackend]) -> bool:
    """List ``flybrain.malecns`` only when extra + data are already present.

    Returns True if registered. Never downloads. Failures are skipped so
    biorouter still serves the default fixtures.
    """
    try:
        from fly_harness.flybrain.backend import FlyBrainBackend
        from fly_harness.flybrain.detect import flybrain_available, flybrain_data_available
    except Exception:
        return False
    if not flybrain_available() or not flybrain_data_available():
        return False
    try:
        backends[FLYBRAIN_MODEL_ID] = FlyBrainBackend.from_installed(
            download=False,
            model_id=FLYBRAIN_MODEL_ID,
        )
    except Exception:
        return False
    return True


def try_register_c302(backends: dict[str, ModelBackend], *, real: bool = False) -> bool:
    """List ``c302.celegans`` when a backend is registered.

    Default (``real=False``): FakeC302 (302 neurons) so CI can prove the id
    without NEURON / Docker / OpenWorm. ``real=True`` only if a running
    tick/reset Model is already in the environment. This repo does not wrap
    NeuroML/NEURON, does not download connectomes, and does not start
    OpenWorm Docker. Failures skip so biorouter still serves fixtures.
    """
    if real:
        return try_register_real_c302(backends)
    try:
        backends[C302_MODEL_ID] = FakeC302(model_id=C302_MODEL_ID)
        return True
    except Exception:
        return False


def try_register_real_c302(backends: dict[str, ModelBackend]) -> bool:
    """Register a real worm Model only if one is already running. Never Docker."""
    try:
        from fly_harness.c302.detect import running_c302_backend
    except Exception:
        return False
    backend = running_c302_backend()
    if backend is None:
        return False
    try:
        backends[getattr(backend, "model_id", C302_MODEL_ID)] = backend
    except Exception:
        return False
    return True


def create_default_backends() -> dict[str, ModelBackend]:
    """LIF reflex fixture + FakeDeployedSim ids + FakeC302; optional flybrain."""
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
    backends: dict[str, ModelBackend] = {
        lif.model_id: lif,
        fake.model_id: fake,
        fake_gain.model_id: fake_gain,
    }
    try_register_c302(backends)
    try_register_flybrain(backends)
    return backends


def create_registry(
    backends: Mapping[str, ModelBackend] | None = None,
) -> BioSimRouter:
    """In-process router used by the HTTP process. Unknown ids error."""
    return BioSimRouter(create_default_backends() if backends is None else backends)
