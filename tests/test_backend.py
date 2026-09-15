"""Tests for swappable ModelBackend docking (LIF, fake sim, router)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from fly_harness import (
    BioSimRouter,
    DirectBioSimBackend,
    FakeDeployedSim,
    FlyHarness,
    HttpModelBackend,
    InProcessLifBackend,
    ModelBackend,
    UnknownModelError,
)
from fly_harness.demo.codec import ReflexDecoder, ReflexEncoder, TouchObservation
from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome


class VectorEncoder:
    def encode(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float64)


class VectorDecoder:
    def decode(self, potentials: np.ndarray) -> np.ndarray:
        return np.asarray(potentials, dtype=np.float64)


def _lif_harness() -> FlyHarness:
    return FlyHarness(
        build_reflex_connectome(),
        ReflexEncoder(),
        ReflexDecoder(),
        sensory_indices=SENSORY_INDICES,
        decay=0.2,
        gain=0.45,
    )


def test_default_lif_backend_still_steps() -> None:
    harness = _lif_harness()
    assert isinstance(harness.backend, InProcessLifBackend)
    assert isinstance(harness.backend, ModelBackend)
    assert harness.model_id == InProcessLifBackend.DEFAULT_MODEL_ID

    harness.step(TouchObservation())
    harness.step(TouchObservation(touch_left=1.0))
    result = harness.step(TouchObservation(touch_left=1.0))
    assert result.action.turn == 1
    assert result.timestamp == 3.0
    assert result.potentials.shape == (24,)


def test_default_lif_reset_rewinds_clock() -> None:
    harness = _lif_harness()
    harness.step(TouchObservation(touch_left=1.0))
    harness.reset()
    assert harness.state.timestamp == 0.0
    assert harness.backend.timestamp == 0.0
    np.testing.assert_allclose(harness.state.potentials, 0.0)


def test_fake_deployed_backend_through_harness() -> None:
    backend = DirectBioSimBackend(
        n_neurons=8,
        model_id="vendor.fake",
        scale=2.0,
    )
    assert isinstance(backend, ModelBackend)
    harness = FlyHarness(encoder=VectorEncoder(), decoder=VectorDecoder(), backend=backend)
    result = harness.step(np.arange(8, dtype=np.float64))
    np.testing.assert_allclose(result.action, 2.0 * np.arange(8, dtype=np.float64))
    assert result.timestamp == 1.0
    assert harness.n_neurons == 8
    assert harness.model_id == "vendor.fake"
    assert harness.state.n_neurons == 8


def test_positional_backend_construction() -> None:
    sim = FakeDeployedSim(4, model_id="pos.fake", scale=0.5)
    harness = FlyHarness(sim, VectorEncoder(), VectorDecoder())
    result = harness.step(np.ones(4))
    np.testing.assert_allclose(result.potentials, 0.5)


def test_router_switches_by_model_id() -> None:
    left = DirectBioSimBackend(
        FakeDeployedSim(6, model_id="sim.left", scale=1.0)
    )
    right = DirectBioSimBackend(
        FakeDeployedSim(6, model_id="sim.right", scale=3.0)
    )
    router = BioSimRouter({"sim.left": left, "sim.right": right}, model_id="sim.left")
    harness = FlyHarness(encoder=VectorEncoder(), decoder=VectorDecoder(), backend=router)

    stimulus = np.ones(6, dtype=np.float64)
    first = harness.step(stimulus)
    np.testing.assert_allclose(first.potentials, 1.0)
    assert harness.model_id == "sim.left"

    router.use("sim.right")
    second = harness.step(stimulus)
    np.testing.assert_allclose(second.potentials, 3.0)
    assert harness.model_id == "sim.right"
    assert router.registered_ids() == ("sim.left", "sim.right")


def test_router_unknown_model_id_errors() -> None:
    fake = DirectBioSimBackend(n_neurons=3, model_id="only.one")
    router = BioSimRouter({"only.one": fake})
    with pytest.raises(UnknownModelError, match="unknown model_id 'missing.sim'"):
        router.use("missing.sim")
    with pytest.raises(UnknownModelError, match="registered: 'only.one'"):
        router.select("nope")
    with pytest.raises(UnknownModelError, match="not a marketplace"):
        router.select("still-missing")


def test_cannot_pass_state_and_backend() -> None:
    backend = DirectBioSimBackend(n_neurons=2, model_id="x")
    with pytest.raises(ValueError, match="not both"):
        FlyHarness(
            build_reflex_connectome(),
            VectorEncoder(),
            VectorDecoder(),
            backend=backend,
        )


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve(self, model_id: str | None) -> Any | None:
        target = self.server.backend  # type: ignore[attr-defined]
        if isinstance(target, BioSimRouter):
            if not model_id:
                self._send(400, {"error": "missing model"})
                return None
            try:
                return target.select(model_id)
            except UnknownModelError:
                self._send(404, {"error": "unknown model_id", "model": model_id})
                return None
        if model_id and model_id != target.model_id:
            self._send(404, {"error": "unknown model_id", "model": model_id})
            return None
        return target

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") != "/status":
            self._send(404, {"error": "not found"})
            return
        model_id = (parse_qs(parsed.query).get("model") or [None])[0]
        inner = self._resolve(model_id)
        if inner is None:
            return
        self._send(
            200,
            {
                "model": inner.model_id,
                "n_neurons": int(inner.n_neurons),
                "timestamp": float(inner.timestamp),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw.decode("utf-8") or "{}")
        model_id = body.get("model")
        inner = self._resolve(model_id)
        if inner is None:
            return
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.endswith("/tick"):
            output = np.asarray(inner.tick(body["input"]), dtype=np.float64)
            self._send(
                200,
                {
                    "model": inner.model_id,
                    "output": output.tolist(),
                    "n_neurons": int(inner.n_neurons),
                    "timestamp": float(inner.timestamp),
                },
            )
            return
        if path.endswith("/reset"):
            pots = body.get("potentials")
            inner.reset(None if pots is None else np.asarray(pots, dtype=np.float64))
            self._send(
                200,
                {
                    "model": inner.model_id,
                    "n_neurons": int(inner.n_neurons),
                    "timestamp": float(inner.timestamp),
                },
            )
            return
        self._send(404, {"error": "not found"})


def _serve(backend: ModelBackend | BioSimRouter) -> tuple[str, ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    server.backend = backend  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return f"http://{host}:{port}", server


def test_http_client_talks_to_router_with_model_field() -> None:
    local = DirectBioSimBackend(n_neurons=5, model_id="remote.sim", scale=4.0)
    router = BioSimRouter({"remote.sim": local})
    url, server = _serve(router)
    try:
        client = HttpModelBackend(url, model_id="remote.sim", n_neurons=5)
        harness = FlyHarness(
            encoder=VectorEncoder(),
            decoder=VectorDecoder(),
            backend=DirectBioSimBackend(url=url, model_id="remote.sim", n_neurons=5),
        )
        result = harness.step(np.ones(5))
        np.testing.assert_allclose(result.potentials, 4.0)
        assert result.timestamp == 1.0
        client.reset()
        assert client.timestamp == 0.0
    finally:
        server.shutdown()
        server.server_close()


def test_http_unknown_model_id_errors() -> None:
    local = DirectBioSimBackend(n_neurons=2, model_id="known")
    router = BioSimRouter({"known": local})
    url, server = _serve(router)
    try:
        client = HttpModelBackend(url, model_id="ghost.sim", n_neurons=2)
        with pytest.raises(UnknownModelError, match="ghost.sim"):
            client.tick(np.zeros(2))
        remote_router = BioSimRouter(remote_url=url)
        forwarded = remote_router.select("ghost.sim")
        assert isinstance(forwarded, HttpModelBackend)
        with pytest.raises(UnknownModelError):
            forwarded.tick(np.zeros(2))
    finally:
        server.shutdown()
        server.server_close()
