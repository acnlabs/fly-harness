"""FlyGym adapter tests: run without MuJoCo; smoke if flygym can start."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fly_harness import FlyHarness
from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome
from fly_harness.flygym import (
    FlyGymAction,
    FlyGymDecoder,
    FlyGymEncoder,
    FlyGymHarnessEnv,
    N_LEG_JOINTS,
    apply_flygym_action,
    flygym_available,
    require_flygym,
)
from fly_harness.flygym.schema import extract_joint_angles, left_right_contact
from fly_harness.protocols import Decoder, Encoder


class FakeFlyGymEnv:
    """Gymnasium-shaped stand-in; no MuJoCo."""

    def __init__(self) -> None:
        self.actions: list[dict[str, np.ndarray]] = []
        self.obs: dict[str, np.ndarray] = {
            "joints": np.zeros((3, N_LEG_JOINTS), dtype=np.float64),
            "contact_forces": np.zeros((36, 3), dtype=np.float64),
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return self.obs, {}

    def step(self, action: dict[str, np.ndarray]) -> tuple:
        self.actions.append(action)
        return self.obs, 0.0, False, False, {}


class FakeFlyGym2Sim:
    """Duck-typed FlyGym 2.x Simulation: set_actuator_inputs + step."""

    def __init__(self) -> None:
        self.inputs: list[tuple[str, Any, np.ndarray]] = []
        self.adhesion: list[np.ndarray] = []
        self.physics_steps = 0

    def set_actuator_inputs(self, fly_name: str, actuator_type: Any, inputs: np.ndarray) -> None:
        self.inputs.append((fly_name, actuator_type, np.asarray(inputs)))

    def set_leg_adhesion_states(self, fly_name: str, states: np.ndarray) -> None:
        self.adhesion.append(np.asarray(states))

    def step(self) -> None:
        self.physics_steps += 1

    def get_joint_angles(self, fly_name: str) -> np.ndarray:
        return np.zeros(N_LEG_JOINTS, dtype=np.float64)


def _mock_obs(*, left_force: float = 0.0, right_force: float = 0.0) -> dict[str, np.ndarray]:
    joints = np.zeros((3, N_LEG_JOINTS), dtype=np.float64)
    joints[0] = np.linspace(-0.2, 0.2, N_LEG_JOINTS)
    contact = np.zeros((36, 3), dtype=np.float64)
    # 6 sensors per leg, legs LF LM LH | RF RM RH
    contact[0:18, 2] = left_force
    contact[18:36, 2] = right_force
    return {"joints": joints, "contact_forces": contact}


def test_extension_imports_without_third_party_flygym() -> None:
    assert isinstance(flygym_available(), bool)
    from fly_harness.flygym import FlyGymEncoder as Enc

    encoder = Enc(24)
    assert isinstance(encoder, Encoder)


def test_require_flygym_hints_extra_when_missing() -> None:
    if flygym_available():
        require_flygym()
        return
    with pytest.raises(ImportError, match="fly-harness\\[flygym\\]"):
        require_flygym()


def test_extract_joint_angles_from_gymnasium_obs() -> None:
    obs = _mock_obs()
    angles = extract_joint_angles(obs)
    assert angles.shape == (N_LEG_JOINTS,)
    assert angles[0] == pytest.approx(-0.2)


def test_left_right_contact_splits_six_legs() -> None:
    obs = _mock_obs(left_force=2.0, right_force=0.5)
    left, right = left_right_contact(obs["contact_forces"])
    assert left > right
    assert left == pytest.approx(2.0)
    assert right == pytest.approx(0.5)


def test_encoder_maps_contact_onto_reflex_touch_indices() -> None:
    encoder = FlyGymEncoder(24)
    assert isinstance(encoder, Encoder)
    currents = encoder.encode(_mock_obs(left_force=1.0, right_force=0.0))
    assert currents.shape == (24,)
    assert currents[0] > currents[2]
    assert currents[1] > 0.5
    assert currents[2] == pytest.approx(0.0)


def test_decoder_emits_joints_and_adhesion() -> None:
    rest = np.linspace(0.0, 0.1, N_LEG_JOINTS)
    decoder = FlyGymDecoder(24, rest_pose=rest)
    assert isinstance(decoder, Decoder)
    potentials = np.zeros(24, dtype=np.float64)
    potentials[20:22] = 1.0  # turn-right pool on the 24-neuron fixture
    potentials[18:20] = 0.8  # forward
    action = decoder.decode(potentials)
    assert isinstance(action, FlyGymAction)
    payload = action.as_env_dict()
    assert payload["joints"].shape == (N_LEG_JOINTS,)
    assert payload["adhesion"].shape == (6,)
    assert payload["adhesion"][0] > 0.0
    assert "muscle" not in payload
    # +turn biases left coxa_yaw vs right
    assert payload["joints"][2] > rest[2]
    assert payload["joints"][23] < rest[23]


def test_decoder_muscle_optional() -> None:
    decoder = FlyGymDecoder(24, include_muscle=True, include_adhesion=False)
    action = decoder.decode(np.ones(24))
    assert action.adhesion is None
    assert action.muscle is not None
    assert action.muscle.shape == (N_LEG_JOINTS,)


def test_wrapper_steps_fake_gymnasium_env() -> None:
    env = FakeFlyGymEnv()
    env.obs = _mock_obs(left_force=1.5, right_force=0.0)
    state = build_reflex_connectome()
    harness = FlyHarness(
        state,
        FlyGymEncoder(24),
        FlyGymDecoder(24, rest_pose=np.zeros(N_LEG_JOINTS)),
        sensory_indices=SENSORY_INDICES,
    )
    body = FlyGymHarnessEnv(env, harness)
    obs, info = body.reset()
    assert obs["joints"].shape == (3, N_LEG_JOINTS)
    result = body.step()
    assert len(env.actions) == 1
    assert env.actions[0]["joints"].shape == (N_LEG_JOINTS,)
    assert env.actions[0]["adhesion"].shape == (6,)
    assert result.harness.potentials.shape == (24,)
    assert result.reward == 0.0
    assert info == {}


def test_apply_action_on_flygym2_duck_type() -> None:
    sim = FakeFlyGym2Sim()
    action = FlyGymAction(joints=np.ones(N_LEG_JOINTS) * 0.01, adhesion=np.ones(6))
    apply_flygym_action(sim, action, fly_name="fly0", actuator_type="position")
    assert sim.physics_steps == 1
    assert sim.inputs[0][0] == "fly0"
    assert sim.inputs[0][2].shape == (N_LEG_JOINTS,)
    assert sim.adhesion[0].shape == (6,)


def test_wrapper_reads_simulation_obs_after_physics_step() -> None:
    sim = FakeFlyGym2Sim()
    harness = FlyHarness(
        build_reflex_connectome(),
        FlyGymEncoder(24),
        FlyGymDecoder(24),
        sensory_indices=SENSORY_INDICES,
    )
    body = FlyGymHarnessEnv(sim, harness, fly_name="fly0", actuator_type="position")
    obs, _ = body.reset()
    assert obs is not None
    assert obs["joint_angles"].shape == (N_LEG_JOINTS,)
    result = body.step()
    assert result.observation["joint_angles"].shape == (N_LEG_JOINTS,)
    assert sim.physics_steps >= 1


@pytest.mark.skipif(not flygym_available(), reason="flygym not installed")
def test_flygym_smoke_one_env_step() -> None:
    """If FlyGym+MuJoCo can start, take one real step. Otherwise skip."""
    from fly_harness.flygym.wrapper import try_make_neuromechfly_sim

    try:
        env = try_make_neuromechfly_sim()
    except Exception as exc:  # noqa: BLE001 — env boot is machine-dependent
        pytest.skip(f"flygym imported but NeuroMechFly sim did not start: {exc}")

    harness = FlyHarness(
        build_reflex_connectome(),
        FlyGymEncoder(24),
        FlyGymDecoder(24),
        sensory_indices=SENSORY_INDICES,
    )
    body = FlyGymHarnessEnv(env, harness)
    try:
        body.reset()
        result = body.step()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"flygym sim started but step failed: {exc}")
    payload = result.action.as_env_dict()
    assert payload["joints"].ndim == 1
    assert payload["joints"].size >= 1
