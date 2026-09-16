"""flybrain extra tests: mock FakeFlyBrain; skip real MaleCNS unless data is on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fly_harness import FlyHarness
from fly_harness.flybrain import (
    DEFAULT_MODEL_ID,
    MALECNS_N_NEURONS,
    FakeFlyBrain,
    FlyBrainBackend,
    FlyBrainInjectEncoder,
    FlyBrainReadoutDecoder,
    flybrain_available,
    flybrain_data_available,
    require_flybrain,
)
from fly_harness.protocols import Decoder, Encoder, ModelBackend


def _core_sources() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "src" / "fly_harness"
    return [
        root / "__init__.py",
        root / "backend.py",
        root / "brain_state.py",
        root / "harness.py",
        root / "http_backend.py",
        root / "protocols.py",
        root / "connectome_loader.py",
    ]


def test_kernel_sources_do_not_import_flybrain() -> None:
    for path in _core_sources():
        text = path.read_text(encoding="utf-8")
        assert "from flybrain" not in text
        assert "import flybrain" not in text
        assert "fly_harness.flybrain" not in text
        assert "caveclient" not in text


def test_extension_imports_without_third_party_flybrain() -> None:
    assert isinstance(flybrain_available(), bool)
    assert isinstance(flybrain_data_available(), bool)
    brain = FakeFlyBrain()
    backend = FlyBrainBackend(brain)
    encoder = FlyBrainInjectEncoder(brain.n, channels={"loom": (0, 1)})
    decoder = FlyBrainReadoutDecoder(brain.n)
    assert isinstance(backend, ModelBackend)
    assert isinstance(encoder, Encoder)
    assert isinstance(decoder, Decoder)
    assert backend.n_neurons == 32
    assert backend.n_neurons != MALECNS_N_NEURONS


def test_require_flybrain_hints_extra_when_missing() -> None:
    if flybrain_available():
        require_flybrain()
        return
    with pytest.raises(ImportError, match="fly-harness\\[flybrain\\]"):
        require_flybrain()


def test_from_installed_does_not_download_when_data_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    if not flybrain_available():
        with pytest.raises(ImportError, match="fly-harness\\[flybrain\\]"):
            FlyBrainBackend.from_installed(download=False)
        return

    monkeypatch.setattr("flybrain.data.has_data", lambda *args, **kwargs: False)
    with pytest.raises(FileNotFoundError, match="will not download"):
        FlyBrainBackend.from_installed(download=False)


def test_fake_backend_through_harness_step() -> None:
    brain = FakeFlyBrain()
    backend = FlyBrainBackend(brain, model_id="flybrain.fake")
    harness = FlyHarness(
        encoder=FlyBrainInjectEncoder(
            brain.n, channels={"loom": tuple(int(i) for i in brain.cells(["sens"]))}
        ),
        decoder=FlyBrainReadoutDecoder(
            brain.n, readout_indices=brain.cells(["descending_neuron"])
        ),
        backend=backend,
    )
    idle = harness.step({"loom": 0.0})
    assert idle.action["n_neurons"] == 32
    assert idle.timestamp == pytest.approx(brain.dt)
    driven = harness.step({"loom": 0.9})
    assert driven.potentials.shape == (32,)
    assert harness.model_id == "flybrain.fake"
    harness.reset()
    assert harness.backend.timestamp == 0.0


def test_tick_injects_nonzero_currents() -> None:
    brain = FakeFlyBrain(n_neurons=8)
    backend = FlyBrainBackend(brain)
    current = np.zeros(8, dtype=np.float64)
    current[2] = 1.5
    out = backend.tick(current)
    assert out.shape == (8,)
    assert backend.last_fired.size >= 1
    assert 2 in backend.last_fired or out[2] == 0.0


def test_example_main_uses_stub(capsys: pytest.CaptureFixture[str]) -> None:
    from fly_harness.demo.flybrain_loop import _BANNER, main

    main(["--steps", "2"])
    out = capsys.readouterr().out
    assert "166,700" in out or "166,700" in _BANNER
    assert "FakeFlyBrain" in out or "stub=True" in out
    assert "160k" in _BANNER
    assert DEFAULT_MODEL_ID.startswith("flybrain")


@pytest.mark.skipif(
    not (flybrain_available() and flybrain_data_available()),
    reason="flybrain not installed or MaleCNS files missing (will not download)",
)
def test_flybrain_smoke_one_tick_without_download() -> None:
    backend = FlyBrainBackend.from_installed(download=False)
    assert backend.n_neurons == MALECNS_N_NEURONS
    assert backend.model_id == DEFAULT_MODEL_ID
    current = np.zeros(backend.n_neurons, dtype=np.float64)
    out = backend.tick(current)
    assert out.shape == (MALECNS_N_NEURONS,)
    backend.reset()
    assert backend.timestamp == 0.0
