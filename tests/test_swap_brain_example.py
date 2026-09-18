"""same loop, swap Model: toy LIF vs flybrain.malecns; skip/mock extras."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fly_harness.backend import DEFAULT_LIF_MODEL_ID
from fly_harness.demo.body_loop import StubFlyGymEnv
from fly_harness.demo.connectome import N_NEURONS
from fly_harness.demo.swap_brain import (
    _BANNER,
    attach_harness,
    harness_for_backend,
    main,
    real_flybrain_ready,
    run_identical_loop,
    run_swap,
    skip_real_b_reason,
    toy_lif_backend,
)
from fly_harness.flybrain import FAKE_N_NEURONS, MALECNS_N_NEURONS
from fly_harness.flygym import N_LEG_JOINTS, FlyGymHarnessEnv
from fly_harness.router import FLYBRAIN_MODEL_ID

ROOT = Path(__file__).resolve().parents[1]


def test_kernel_sources_do_not_import_vendor_models() -> None:
    src = ROOT / "src" / "fly_harness"
    for name in ("__init__.py", "harness.py", "backend.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "fly_harness.flybrain" not in text
        assert "fly_harness.flygym" not in text
        assert "from flybrain" not in text
        assert "import flybrain" not in text
        assert "demo.swap_brain" not in text
        assert "FlyBrainBackend" not in text


def test_swap_reuses_same_body_and_loop() -> None:
    env = StubFlyGymEnv()
    result = run_swap(steps=3, env=env)
    assert result.same_body is True
    assert result.stub_body is True
    assert result.body_name == "StubFlyGymEnv"
    assert result.backend_a.skipped is False
    assert result.backend_b.skipped is False
    assert result.backend_a.model_id == DEFAULT_LIF_MODEL_ID == "fly-harness.in-process-lif"
    assert result.backend_a.n_neurons == N_NEURONS == 24
    assert result.backend_b.model_id == FLYBRAIN_MODEL_ID == "flybrain.malecns"
    assert result.backend_b.n_neurons == FAKE_N_NEURONS == 32
    assert result.backend_b.n_neurons != MALECNS_N_NEURONS
    assert result.backend_b.stub_model is True
    assert len(result.backend_a.traces) == 3
    assert len(result.backend_b.traces) == 3
    assert [row["model_id"] for row in result.backend_a.traces] == [DEFAULT_LIF_MODEL_ID] * 3
    assert [row["model_id"] for row in result.backend_b.traces] == [FLYBRAIN_MODEL_ID] * 3
    joints = np.asarray(result.backend_a.traces[1]["joints"])
    assert joints.shape == (N_LEG_JOINTS,)
    assert len(env.actions) == 6  # 3 steps × two Models, same env


def test_identical_loop_function_is_shared() -> None:
    env = StubFlyGymEnv()
    body = FlyGymHarnessEnv(env, harness_for_backend(toy_lif_backend()))
    traces_a = run_identical_loop(body, steps=2)
    attach_harness(body, harness_for_backend(toy_lif_backend()))
    traces_a2 = run_identical_loop(body, steps=2)
    assert [row["model_id"] for row in traces_a] == [DEFAULT_LIF_MODEL_ID] * 2
    assert traces_a2[0]["n_neurons"] == 24
    assert run_identical_loop.__doc__ is not None
    assert "obs → FlyHarness.step" in run_identical_loop.__doc__


def test_main_default_prints_both_model_ids(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--steps", "1"])
    out = capsys.readouterr().out
    assert "same loop, swap Model" in out
    assert DEFAULT_LIF_MODEL_ID in out
    assert FLYBRAIN_MODEL_ID in out
    assert "does not claim better walking" in _BANNER
    assert "not a harness feature" in _BANNER
    assert "motor CPG" in _BANNER
    assert "brain-level" in _BANNER
    assert "will not pretend" in _BANNER.lower() or "Will not pretend" in _BANNER
    assert "166,700" in _BANNER
    assert "consciousness" in _BANNER.lower()
    assert "same loop, two model_ids" in out
    assert "24" in out
    assert "32" in out


def test_main_real_runs_a_and_skips_b_without_data(capsys: pytest.CaptureFixture[str]) -> None:
    if real_flybrain_ready():
        pytest.skip("flybrain extra + MaleCNS files are present")
    main(["--real", "--steps", "1"])
    out = capsys.readouterr().out
    assert DEFAULT_LIF_MODEL_ID in out
    assert "SKIPPED" in out
    assert "Will not download" in out
    assert str(MALECNS_N_NEURONS) in out
    assert "Will not pretend" in out
    assert "B skipped honestly" in out
    assert skip_real_b_reason().split(":")[0] in out or "skip B" in out


def test_pyproject_exposes_script_and_keeps_0_5_4() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.5.4"' in text
    assert "fly-harness-swap-brain-demo" in text
    assert "fly_harness.demo.swap_brain:main" in text


def test_readme_and_usage_point_at_same_loop_swap() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
    assert "same loop, swap Model" in readme
    assert "examples/swap_brain.py" in readme
    assert "does not claim better walking" in readme.lower() or "not claim better walking" in readme
    assert "### 4." not in readme
    assert "same loop, swap Model" in usage
    assert "examples/swap_brain.py" in usage
    assert "\n## 4." not in usage
    assert "Will not download" in usage or "will not download" in usage.lower()
    assert "166,700" in usage
    assert "motor CPG" in usage
    assert "brain-level" in usage
    assert "not the harness kernel" in usage.replace("**", "")
    assert "motor CPG" in readme
    assert "brain-level" in readme


@pytest.mark.skipif(
    not real_flybrain_ready(),
    reason="flybrain not installed or MaleCNS files missing (will not download)",
)
def test_real_malecns_swap_without_download() -> None:
    result = run_swap(steps=1, real=True)
    assert result.backend_a.model_id == DEFAULT_LIF_MODEL_ID
    assert result.backend_b.skipped is False
    assert result.backend_b.n_neurons == MALECNS_N_NEURONS
    assert result.backend_b.model_id == FLYBRAIN_MODEL_ID
    assert result.backend_b.stub_model is False
