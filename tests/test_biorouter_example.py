"""biorouter example: in-thread stdlib server, no live CLI daemon, no FlyWire."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fly_harness.demo.biorouter_loop import (
    FIXTURE_FAKE_N_NEURONS,
    _BANNER,
    build_loop,
    iter_inprocess_loop,
    main,
    run_scripted_loop,
)
from fly_harness.router.registry import DEFAULT_FAKE_GAIN_MODEL_ID, DEFAULT_FAKE_MODEL_ID


def test_inprocess_loop_stays_on_fixture_size() -> None:
    for loop in iter_inprocess_loop():
        assert loop.n_neurons == FIXTURE_FAKE_N_NEURONS == 8
        assert loop.n_neurons != 166_000
        assert loop.model_id == DEFAULT_FAKE_MODEL_ID


def test_scripted_loop_ticks_switches_and_unknown() -> None:
    for loop in iter_inprocess_loop():
        traces = run_scripted_loop(loop, steps=2)
        direct = [row for row in traces if row["phase"] == "direct"]
        switched = [row for row in traces if row["phase"] == "switch"]
        unknown = [row for row in traces if row["phase"] == "unknown"]
        assert len(direct) == 2
        assert direct[0]["n_neurons"] == 8
        np.testing.assert_allclose(direct[0]["potentials"], 1.0)
        assert switched[0]["model_id"] == DEFAULT_FAKE_GAIN_MODEL_ID
        np.testing.assert_allclose(switched[0]["potentials"], 3.0)
        assert unknown[0]["error"] == "UnknownModelError"
        assert unknown[0]["model_id"] == "ghost.sim"


def test_kernel_init_does_not_import_biorouter_example() -> None:
    init_text = (
        Path(__file__).resolve().parents[1] / "src" / "fly_harness" / "__init__.py"
    ).read_text(encoding="utf-8")
    assert "demo.biorouter_loop" not in init_text
    assert "fly_harness.router" not in init_text


def test_main_inprocess_mentions_positioning(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--steps", "1"])
    out = capsys.readouterr().out
    assert "OpenRouter routes existing LLMs" in out or "OpenRouter" in _BANNER
    assert "biorouter" in out
    assert "166k" in out
    assert "marketplace" in out.lower() or "marketplace" in _BANNER.lower()
    assert "8" in out


@pytest.mark.skipif(
    not os.environ.get("BIOROUTER_URL"),
    reason="no live biorouter; set BIOROUTER_URL to attach",
)
def test_attach_to_external_biorouter() -> None:
    url = os.environ["BIOROUTER_URL"]
    loop = build_loop(url)
    traces = run_scripted_loop(loop, steps=1)
    assert traces[0]["n_neurons"] == FIXTURE_FAKE_N_NEURONS
