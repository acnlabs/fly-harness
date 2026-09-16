"""biorouter → c302.celegans example: FakeC302 in-thread; skip real OpenWorm."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fly_harness.backend import UnknownModelError
from fly_harness.c302 import (
    C302_MODEL_ID,
    C302_N_NEURONS,
    FakeC302,
    c302_available,
    running_c302_backend,
)
from fly_harness.demo.biorouter_c302_loop import (
    _BANNER,
    build_loop,
    iter_inprocess_loop,
    main,
    run_scripted_loop,
)
from fly_harness.flybrain import FAKE_N_NEURONS, FakeFlyBrain, FlyBrainBackend
from fly_harness.http_backend import HttpModelBackend
from fly_harness.router import FLYBRAIN_MODEL_ID, create_default_backends, create_registry
from fly_harness.router.server import running_router


def test_kernel_init_does_not_import_c302_or_openworm() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "fly_harness"
    init_text = (root / "__init__.py").read_text(encoding="utf-8")
    harness_text = (root / "harness.py").read_text(encoding="utf-8")
    backend_text = (root / "backend.py").read_text(encoding="utf-8")
    for text in (init_text, harness_text, backend_text):
        assert "demo.biorouter_c302_loop" not in text
        assert "fly_harness.c302" not in text
        assert "FakeC302" not in text
        assert "from c302" not in text
        assert "import c302" not in text
        assert "openworm" not in text.lower()
        assert "OpenWorm" not in text


def test_pyproject_has_no_c302_or_openworm_extra() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "fly-harness-biorouter-c302-demo" in text
    assert "fly_harness.demo.biorouter_c302_loop:main" in text
    assert "c302 =" not in text
    assert "openworm =" not in text
    assert 'version = "0.5.2"' in text


def test_inprocess_loop_lists_fake_c302() -> None:
    for loop in iter_inprocess_loop():
        assert loop.model_id == C302_MODEL_ID == "c302.celegans"
        assert loop.n_neurons == C302_N_NEURONS == 302
        assert loop.stub is True
        assert isinstance(FakeC302(), FakeC302)


def test_scripted_loop_ticks_c302_id_and_unknown_404() -> None:
    for loop in iter_inprocess_loop():
        traces = run_scripted_loop(loop, steps=2)
        ticks = [row for row in traces if row["phase"] == "c302"]
        unknown = [row for row in traces if row["phase"] == "unknown"]
        assert len(ticks) == 2
        assert ticks[0]["model_id"] == C302_MODEL_ID
        assert ticks[0]["n_neurons"] == 302
        assert ticks[0]["action"]["n_neurons"] == 302
        assert ticks[1]["timestamp"] > ticks[0]["timestamp"]
        assert unknown[0]["error"] == "UnknownModelError"
        assert unknown[0]["model_id"] == "ghost.sim"


def test_unlisted_c302_celegans_on_plain_router_is_404() -> None:
    backends = create_default_backends()
    backends.pop(C302_MODEL_ID, None)
    registry = create_registry(backends)
    assert C302_MODEL_ID not in registry.registered_ids()
    with running_router(registry=registry) as (url, _server):
        client = HttpModelBackend(url, model_id=C302_MODEL_ID, n_neurons=8)
        with pytest.raises(UnknownModelError, match="c302.celegans"):
            client.tick(np.zeros(8))


def test_two_model_ids_fake_flybrain_and_fake_c302() -> None:
    backends = create_default_backends()
    backends[FLYBRAIN_MODEL_ID] = FlyBrainBackend(
        FakeFlyBrain(), model_id=FLYBRAIN_MODEL_ID
    )
    registry = create_registry(backends)
    ids = registry.registered_ids()
    assert FLYBRAIN_MODEL_ID in ids
    assert C302_MODEL_ID in ids
    with running_router(registry=registry) as (url, _server):
        fly = HttpModelBackend(
            url, model_id=FLYBRAIN_MODEL_ID, n_neurons=FAKE_N_NEURONS
        )
        worm = HttpModelBackend(url, model_id=C302_MODEL_ID, n_neurons=C302_N_NEURONS)
        fly_out = fly.tick(np.zeros(FAKE_N_NEURONS, dtype=np.float64))
        worm_current = np.zeros(C302_N_NEURONS, dtype=np.float64)
        worm_current[:4] = 0.5
        worm_out = worm.tick(worm_current)
        assert fly.model_id == FLYBRAIN_MODEL_ID
        assert fly_out.shape == (FAKE_N_NEURONS,)
        assert worm.model_id == C302_MODEL_ID
        assert worm_out.shape == (C302_N_NEURONS,)
        np.testing.assert_allclose(worm_out[:4], 0.5)
        ghost = HttpModelBackend(url, model_id="ghost.sim", n_neurons=8)
        with pytest.raises(UnknownModelError, match="ghost.sim"):
            ghost.tick(np.zeros(8))


def test_main_inprocess_mentions_positioning(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--steps", "1"])
    out = capsys.readouterr().out
    assert "c302.celegans" in out
    assert "FakeC302" in out or "stub=True" in out
    assert "302" in out
    assert "hermaphrodite" in _BANNER.lower()
    assert "not a harness feature" in _BANNER
    assert "hosted OpenWorm" in _BANNER
    assert "consciousness" in _BANNER.lower()
    assert "fly-harness[c302]" in _BANNER


def test_main_real_exits_without_importable_c302() -> None:
    if c302_available() and running_c302_backend() is not None:
        pytest.skip("a running c302 Model is present")
    with pytest.raises(SystemExit, match="OpenWorm Docker"):
        main(["--real"])


def test_main_rejects_real_with_url() -> None:
    with pytest.raises(SystemExit, match="in-thread registry"):
        main(["--url", "http://127.0.0.1:8765", "--real"])


@pytest.mark.skipif(
    not os.environ.get("BIOROUTER_URL"),
    reason="no live biorouter; set BIOROUTER_URL to attach",
)
def test_attach_to_external_biorouter() -> None:
    url = os.environ["BIOROUTER_URL"]
    loop = build_loop(url)
    traces = run_scripted_loop(loop, steps=1)
    assert traces[0]["model_id"] == C302_MODEL_ID


@pytest.mark.skipif(
    running_c302_backend() is None,
    reason="no running c302 Model (will not download, will not start OpenWorm Docker)",
)
def test_real_c302_through_inprocess_biorouter() -> None:
    for loop in iter_inprocess_loop(real=True):
        assert loop.model_id == C302_MODEL_ID
        traces = run_scripted_loop(loop, steps=1)
        assert traces[0]["model_id"] == C302_MODEL_ID
