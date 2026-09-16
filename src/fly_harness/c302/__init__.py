"""C. elegans listing stand-in: a Model id, not a fly-harness extra.

Switching species is: deploy a running sim + list an id on biorouter.
``c302.celegans`` is that id. Default CI uses ``FakeC302`` (302 hermaphrodite
neurons) so tests do not need NEURON, Docker, or OpenWorm.

Not ``fly-harness[c302]`` / ``fly-harness[openworm]``. Not a new whole-brain
PyPI package. Not a hosted OpenWorm API. Not a consciousness claim.
Core (``__init__`` / ``harness`` / ``backend``) does not import this module.
"""

from fly_harness.c302.detect import c302_available, running_c302_backend
from fly_harness.c302.fake import C302_MODEL_ID, C302_N_NEURONS, FakeC302

__all__ = [
    "C302_MODEL_ID",
    "C302_N_NEURONS",
    "FakeC302",
    "c302_available",
    "running_c302_backend",
]
