"""Optional FlyGym / NeuroMechFly body adapter.

This package is not the fly-harness core. Core stays numpy/scipy-only.
Install the extra with ``pip install 'fly-harness[flygym]'``.

Encoder/Decoder work on FlyGym observation/action dicts without importing
MuJoCo. The thin wrapper talks to a caller-built env.
"""

from fly_harness.flygym.decoder import FlyGymDecoder
from fly_harness.flygym.detect import flygym_available, import_flygym_module, require_flygym
from fly_harness.flygym.encoder import FlyGymEncoder
from fly_harness.flygym.schema import (
    N_LEG_JOINTS,
    N_LEGS,
    FlyGymAction,
    coerce_action,
    extract_contact_forces,
    extract_joint_angles,
)
from fly_harness.flygym.wrapper import (
    BodyStepResult,
    FlyGymHarnessEnv,
    apply_flygym_action,
    try_make_neuromechfly_sim,
)

__all__ = [
    "BodyStepResult",
    "FlyGymAction",
    "FlyGymDecoder",
    "FlyGymEncoder",
    "FlyGymHarnessEnv",
    "N_LEG_JOINTS",
    "N_LEGS",
    "apply_flygym_action",
    "coerce_action",
    "extract_contact_forces",
    "extract_joint_angles",
    "flygym_available",
    "import_flygym_module",
    "require_flygym",
    "try_make_neuromechfly_sim",
]
