"""body-loop example: stub FlyGym + fake flybrain.malecns; skip real extras."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fly_harness.demo.body_loop import (
    _BANNER,
    StubFlyGymEnv,
    contact_observation,
    iter_inprocess_loop,
    main,
    run_scripted_loop,
)
from fly_harness.flybrain import FAKE_N_NEURONS, MALECNS_N_NEURONS, flybrain_available, flybrain_data_available
from fly_harness.flygym import N_LEG_JOINTS, flygym_available
from fly_harness.router import FLYBRAIN_MODEL_ID


def test_kernel_sources_do_not_import_flygym_or_flybrain() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "fly_harness"
    for name in ("__init__.py", "harness.py", "backend.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "fly_harness.flybrain" not in text
        assert "fly_harness.flygym" not in text
        assert "from flybrain" not in text
        assert "import flybrain" not in text
        assert "FlyBrainBackend" not in text
        assert "demo.body_loop" not in text


def test_inprocess_loop_uses_stub_body_and_fake_model() -> None:
    for loop in iter_inprocess_loop():
        assert loop.model_id == FLYBRAIN_MODEL_ID == "flybrain.malecns"
        assert loop.n_neurons == FAKE_N_NEURONS == 32
        assert loop.n_neurons != MALECNS_N_NEURONS
        assert loop.stub_model is True
        assert loop.stub_body is True
        assert isinstance(loop.body.env, StubFlyGymEnv)


def test_scripted_loop_applies_flygym_action_and_unknown_404() -> None:
    for loop in iter_inprocess_loop():
        traces = run_scripted_loop(loop, steps=3)
        body = [row for row in traces if row["phase"] == "body"]
        unknown = [row for row in traces if row["phase"] == "unknown"]
        assert len(body) == 3
        assert body[0]["model_id"] == FLYBRAIN_MODEL_ID
        assert body[0]["n_neurons"] == 32
        joints = np.asarray(body[1]["joints"])
        assert joints.shape == (N_LEG_JOINTS,)
        adhesion = np.asarray(body[1]["adhesion"])
        assert adhesion.shape == (6,)
        assert unknown[0]["error"] == "UnknownModelError"
        assert unknown[0]["model_id"] == "ghost.sim"
        env = loop.body.env
        assert isinstance(env, StubFlyGymEnv)
        assert len(env.actions) == 3
        assert env.actions[0]["joints"].shape == (N_LEG_JOINTS,)


def test_contact_observation_is_flygym_shaped() -> None:
    obs = contact_observation(touch_left=1.0, touch_right=0.0)
    assert obs["joints"].shape == (3, N_LEG_JOINTS)
    assert obs["contact_forces"].shape == (36, 3)
    assert float(obs["contact_forces"][0, 2]) == 1.0
    assert float(obs["contact_forces"][-1, 2]) == 0.0


def test_main_inprocess_mentions_positioning(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--steps", "1"])
    out = capsys.readouterr().out
    assert "flybrain.malecns" in out
    assert "StubFlyGymEnv" in out or "stub_body=True" in out
    assert "32" in out
    assert "160k" in _BANNER
    assert "not a harness feature" in _BANNER
    assert "consciousness" in _BANNER.lower()


def test_main_real_exits_without_data() -> None:
    if flybrain_available() and flybrain_data_available():
        pytest.skip("flybrain extra + MaleCNS files are present")
    with pytest.raises(SystemExit, match="Will not download"):
        main(["--real"])


def test_main_rejects_real_with_url() -> None:
    with pytest.raises(SystemExit, match="in-thread registry"):
        main(["--url", "http://127.0.0.1:8765", "--real"])


def test_pyproject_exposes_example_script_and_keeps_0_5_5() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'version = "0.5.5"' in text
    assert "fly-harness-body-loop-demo" in text
    assert "fly_harness.demo.body_loop:main" in text


def test_try_flygym_falls_back_to_stub_without_mujoco() -> None:
    if flygym_available():
        pytest.skip("flygym is installed; fallback path is for CI without MuJoCo")
    for loop in iter_inprocess_loop(try_real_flygym=True):
        assert loop.stub_body is True
        assert isinstance(loop.body.env, StubFlyGymEnv)


@pytest.mark.skipif(
    not (flybrain_available() and flybrain_data_available()),
    reason="flybrain not installed or MaleCNS files missing (will not download)",
)
def test_real_malecns_body_loop_without_download() -> None:
    for loop in iter_inprocess_loop(real=True):
        assert loop.n_neurons == MALECNS_N_NEURONS
        assert loop.model_id == FLYBRAIN_MODEL_ID
        traces = run_scripted_loop(loop, steps=1)
        assert traces[0]["n_neurons"] == MALECNS_N_NEURONS
