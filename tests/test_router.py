"""Optional local HTTP ModelBackend router — stdlib, no FastAPI extra required."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fly_harness import (
    BioSimRouter,
    DirectBioSimBackend,
    FlyHarness,
    HttpModelBackend,
    UnknownModelError,
)
from fly_harness.backend import DEFAULT_LIF_MODEL_ID
from fly_harness.router import (
    DEFAULT_FAKE_GAIN_MODEL_ID,
    DEFAULT_FAKE_MODEL_ID,
    DEFAULT_FAKE_N_NEURONS,
    DEFAULT_HOST,
    create_registry,
    running_router,
)
from fly_harness.router.server import build_arg_parser


class VectorEncoder:
    def encode(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float64)


class VectorDecoder:
    def decode(self, potentials: np.ndarray) -> np.ndarray:
        return np.asarray(potentials, dtype=np.float64)


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


def test_kernel_sources_do_not_import_router_or_fastapi() -> None:
    for path in _core_sources():
        text = path.read_text(encoding="utf-8")
        assert "fly_harness.router" not in text
        assert "fastapi" not in text.lower()
        assert "from fastapi" not in text
        assert "http.server" not in text


def test_default_registry_has_lif_and_fake_ids() -> None:
    registry = create_registry()
    ids = registry.registered_ids()
    assert DEFAULT_LIF_MODEL_ID in ids
    assert DEFAULT_FAKE_MODEL_ID in ids
    assert DEFAULT_FAKE_GAIN_MODEL_ID in ids
    assert registry.select(DEFAULT_LIF_MODEL_ID).n_neurons == 24
    assert registry.select(DEFAULT_FAKE_MODEL_ID).n_neurons == DEFAULT_FAKE_N_NEURONS


def test_cli_defaults_bind_localhost() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.host == DEFAULT_HOST == "127.0.0.1"
    assert args.port == 8765


def test_http_client_ticks_fake_through_local_router() -> None:
    with running_router() as (url, _server):
        client = HttpModelBackend(
            url, model_id=DEFAULT_FAKE_MODEL_ID, n_neurons=DEFAULT_FAKE_N_NEURONS
        )
        output = client.tick(np.arange(DEFAULT_FAKE_N_NEURONS, dtype=np.float64))
        np.testing.assert_allclose(output, np.arange(DEFAULT_FAKE_N_NEURONS, dtype=np.float64))
        assert client.timestamp == 1.0
        client.reset()
        assert client.timestamp == 0.0


def test_flyharness_steps_via_direct_url() -> None:
    with running_router() as (url, _server):
        backend = DirectBioSimBackend(
            url=url,
            model_id=DEFAULT_FAKE_MODEL_ID,
            n_neurons=DEFAULT_FAKE_N_NEURONS,
        )
        harness = FlyHarness(
            encoder=VectorEncoder(),
            decoder=VectorDecoder(),
            backend=backend,
        )
        stimulus = np.ones(DEFAULT_FAKE_N_NEURONS, dtype=np.float64)
        result = harness.step(stimulus)
        np.testing.assert_allclose(result.potentials, 1.0)
        assert result.timestamp == 1.0
        assert harness.model_id == DEFAULT_FAKE_MODEL_ID
        harness.reset()
        assert harness.backend.timestamp == 0.0


def test_flyharness_steps_lif_fixture_via_http() -> None:
    with running_router() as (url, _server):
        backend = HttpModelBackend(url, model_id=DEFAULT_LIF_MODEL_ID, n_neurons=24)
        harness = FlyHarness(
            encoder=VectorEncoder(),
            decoder=VectorDecoder(),
            backend=backend,
        )
        result = harness.step(np.zeros(24, dtype=np.float64))
        assert result.potentials.shape == (24,)
        assert result.timestamp == 1.0
        assert harness.n_neurons == 24


def test_router_switches_models_over_http() -> None:
    with running_router() as (url, _server):
        remote = BioSimRouter(remote_url=url, model_id=DEFAULT_FAKE_MODEL_ID)
        harness = FlyHarness(
            encoder=VectorEncoder(),
            decoder=VectorDecoder(),
            backend=remote,
        )
        stimulus = np.ones(DEFAULT_FAKE_N_NEURONS, dtype=np.float64)
        first = harness.step(stimulus)
        np.testing.assert_allclose(first.potentials, 1.0)
        assert harness.model_id == DEFAULT_FAKE_MODEL_ID

        remote.use(DEFAULT_FAKE_GAIN_MODEL_ID)
        second = harness.step(stimulus)
        np.testing.assert_allclose(second.potentials, 3.0)
        assert harness.model_id == DEFAULT_FAKE_GAIN_MODEL_ID


def test_unknown_model_id_is_404_and_unknown_model_error() -> None:
    with running_router() as (url, _server):
        client = HttpModelBackend(url, model_id="ghost.sim", n_neurons=8)
        with pytest.raises(UnknownModelError, match="ghost.sim"):
            client.tick(np.zeros(8))
        with pytest.raises(UnknownModelError, match="ghost.sim"):
            client.reset()
        status_client = HttpModelBackend(url, model_id="also.missing")
        with pytest.raises(UnknownModelError, match="also.missing"):
            _ = status_client.n_neurons

        forwarded = BioSimRouter(remote_url=url).select("no-such-model")
        assert isinstance(forwarded, HttpModelBackend)
        with pytest.raises(UnknownModelError):
            forwarded.tick(np.zeros(8))
