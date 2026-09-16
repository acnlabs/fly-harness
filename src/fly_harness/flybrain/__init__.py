"""Optional flybrain Model extra: dock a running MaleCNS LIF sim.

Third-party package: https://pypi.org/project/flybrain/
(https://github.com/alextitonis/fly.ai). Core stays numpy/scipy-only.

This extra docks a **running** sim. Google/Janelia released a dump, not a
hosted sim. Not the FlyWire website, not a consciousness/upload claim, and
not ~160k "parameters" — MaleCNS v1.0 is **166,700 neurons**.
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
