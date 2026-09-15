"""fly_harness: a minimal microkernel for sparse fly-inspired neural harnesses."""

from fly_harness.brain_state import BrainState
from fly_harness.connectome_loader import ConnectomeLoadResult, load_connectome
from fly_harness.harness import FlyHarness
from fly_harness.protocols import Decoder, Encoder

__all__ = [
    "BrainState",
    "ConnectomeLoadResult",
    "Decoder",
    "Encoder",
    "FlyHarness",
    "load_connectome",
]
__version__ = "0.1.1"
