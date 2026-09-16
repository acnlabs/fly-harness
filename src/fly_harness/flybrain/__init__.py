"""Adapter so **biorouter** / examples can talk to third-party ``flybrain``.

Not a harness feature: the core stays a model-agnostic ``ModelBackend`` port.
Google/Janelia released a dump, not a hosted sim. Not the FlyWire website,
not a consciousness/upload claim, and not ~160k "parameters" — MaleCNS v1.0
is **166,700 neurons**.
"""

from fly_harness.flybrain.backend import (
    DEFAULT_MODEL_ID,
    MALECNS_N_NEURONS,
    FlyBrainBackend,
)
from fly_harness.flybrain.codec import FlyBrainInjectEncoder, FlyBrainReadoutDecoder
from fly_harness.flybrain.detect import (
    flybrain_available,
    flybrain_data_available,
    require_flybrain,
)
from fly_harness.flybrain.fake import FAKE_N_NEURONS, FakeFlyBrain

__all__ = [
    "DEFAULT_MODEL_ID",
    "FAKE_N_NEURONS",
    "MALECNS_N_NEURONS",
    "FakeFlyBrain",
    "FlyBrainBackend",
    "FlyBrainInjectEncoder",
    "FlyBrainReadoutDecoder",
    "flybrain_available",
    "flybrain_data_available",
    "require_flybrain",
]
