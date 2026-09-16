"""biorouter → flybrain.malecns example: fake in-thread; skip real MaleCNS."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fly_harness.backend import UnknownModelError
from fly_harness.demo.biorouter_flybrain_loop import (
    _BANNER,
    build_loop,
    iter_inprocess_loop,
    main,
    run_scripted_loop,
)
from fly_harness.flybrain import (
    FAKE_N_NEURONS,
    MALECNS_N_NEURONS,
    flybrain_available,
    flybrain_data_available,
)
from fly_harness.http_backend import HttpModelBackend
from fly_harness.router import FLYBRAIN_MODEL_ID, create_default_backends, create_registry
from fly_harness.router.server import running_router


def test_kernel_init_does_not_import_example() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "fly_harness"
    init_text = (root / "__init__.py").read_text(encoding="utf-8")
    harness_text = (root / "harness.py").read_text(encoding="utf-8")
    backend_text = (root / "backend.py").read_text(encoding="utf-8")
    assert "demo.biorouter_flybrain_loop" not in init_text
    assert "FlyBrainBackend" not in init_text
    assert "fly_harness.flybrain" not in harness_text
    assert "fly_harness.flybrain" not in backend_text


def test_inprocess_loop_lists_fake_flybrain_malecns() -> None:
    for loop in iter_inprocess_loop():
        assert loop.model_id == FLYBRAIN_MODEL_ID == "flybrain.malecns"
        assert loop.n_neurons == FAKE_N_NEURONS == 32
        assert loop.n_neurons != MALECNS_N_NEURONS
        assert loop.stub is True


def test_scripted_loop_ticks_flybrain_id_and_unknown_404() -> None:
    for loop in iter_inprocess_loop():
        traces = run_scripted_loop(loop, steps=2)
        ticks = [row for row in traces if row["phase"] == "flybrain"]
        unknown = [row for row in traces if row["phase"] == "unknown"]
        assert len(ticks) == 2
        assert ticks[0]["model_id"] == FLYBRAIN_MODEL_ID
        assert ticks[0]["n_neurons"] == 32
        assert ticks[0]["action"]["n_neurons"] == 32
        assert ticks[1]["timestamp"] > ticks[0]["timestamp"]
        assert unknown[0]["error"] == "UnknownModelError"
        assert unknown[0]["model_id"] == "ghost.sim"


def test_unlisted_flybrain_malecns_on_plain_router_is_404() -> None:
    backends = create_default_backends()
    backends.pop(FLYBRAIN_MODEL_ID, None)
    registry = create_registry(backends)
    assert FLYBRAIN_MODEL_ID not in registry.registered_ids()
    with running_router(registry=registry) as (url, _server):
        client = HttpModelBackend(url, model_id=FLYBRAIN_MODEL_ID, n_neurons=8)
        with pytest.raises(UnknownModelError, match="flybrain.malecns"):
            client.tick(np.zeros(8))


def test_main_inprocess_mentions_positioning(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--steps", "1"])
    out = capsys.readouterr().out
    assert "flybrain.malecns" in out
    assert "FakeFlyBrain" in out or "stub=True" in out
    assert "32" in out
    assert "160k" in _BANNER
    assert "not a harness feature" in _BANNER
    assert "marketplace" in _BANNER.lower()
    assert "166,700" in _BANNER or "166,700" in out


def test_main_real_exits_without_data() -> None:
    if flybrain_available() and flybrain_data_available():
        pytest.skip("flybrain extra + MaleCNS files are present")
    with pytest.raises(SystemExit, match="Will not download"):
        main(["--real"])


def test_main_rejects_real_with_url() -> None:
    with pytest.raises(SystemExit, match="in-thread registry"):
        main(["--url", "http://127.0.0.1:8765", "--real"])


def test_pyproject_exposes_example_script() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "fly-harness-biorouter-flybrain-demo" in text
    assert "fly_harness.demo.biorouter_flybrain_loop:main" in text


@pytest.mark.skipif(
    not os.environ.get("BIOROUTER_URL"),
    reason="no live biorouter; set BIOROUTER_URL to attach",
)
def test_attach_to_external_biorouter() -> None:
    url = os.environ["BIOROUTER_URL"]
    loop = build_loop(url)
    traces = run_scripted_loop(loop, steps=1)
    assert traces[0]["model_id"] == FLYBRAIN_MODEL_ID


@pytest.mark.skipif(
    not (flybrain_available() and flybrain_data_available()),
    reason="flybrain not installed or MaleCNS files missing (will not download)",
)
def test_real_malecns_through_inprocess_biorouter() -> None:
    for loop in iter_inprocess_loop(real=True):
        assert loop.n_neurons == MALECNS_N_NEURONS
        assert loop.model_id == FLYBRAIN_MODEL_ID
        traces = run_scripted_loop(loop, steps=1)
        assert traces[0]["n_neurons"] == MALECNS_N_NEURONS
        assert traces[0]["model_id"] == FLYBRAIN_MODEL_ID
