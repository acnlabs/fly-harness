"""Stdlib HTTP process matching ``HttpModelBackend`` (POST /tick, /reset, GET /status).

Public CLI is ``biorouter`` (``python -m fly_harness.router`` is the same
entry). Bind ``127.0.0.1`` by default. OpenRouter routes existing LLMs;
biorouter routes existing deployed biological simulation models. Not the
harness, not FlyWire dumps, not a marketplace.

No FastAPI. Core (``BrainState`` / ``FlyHarness``) does not import this module.
"""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

import numpy as np

from fly_harness.backend import BioSimRouter, UnknownModelError
from fly_harness.router.registry import create_registry

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class RouterHTTPServer(ThreadingHTTPServer):
    """Threading server holding the in-process backend registry."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        registry: BioSimRouter,
    ) -> None:
        super().__init__(server_address, RouterRequestHandler)
        self.registry = registry
        self._lock = threading.RLock()


class RouterRequestHandler(BaseHTTPRequestHandler):
    """JSON endpoints with an OpenRouter-shaped ``model`` field."""

    server: RouterHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return None
        if not isinstance(parsed, dict):
            self._send(400, {"error": "json object required"})
            return None
        return parsed

    def _resolve(self, model_id: str | None) -> Any | None:
        if not model_id:
            self._send(400, {"error": "missing model"})
            return None
        try:
            with self.server._lock:
                return self.server.registry.select(str(model_id))
        except UnknownModelError:
            self._send(404, {"error": "unknown model_id", "model": model_id})
            return None

    def _snapshot(self, inner: Any) -> dict[str, Any]:
        return {
            "model": inner.model_id,
            "n_neurons": int(inner.n_neurons),
            "timestamp": float(inner.timestamp),
        }

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") != "/status":
            self._send(404, {"error": "not found"})
            return
        model_id = (parse_qs(parsed.query).get("model") or [None])[0]
        inner = self._resolve(model_id)
        if inner is None:
            return
        with self.server._lock:
            self._send(200, self._snapshot(inner))

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_json()
        if body is None:
            return
        model_id = body.get("model")
        inner = self._resolve(model_id)
        if inner is None:
            return
        path = urlparse(self.path).path.rstrip("/")
        try:
            with self.server._lock:
                if path.endswith("/tick"):
                    current = np.asarray(body.get("input"), dtype=np.float64)
                    output = np.asarray(inner.tick(current), dtype=np.float64)
                    payload = self._snapshot(inner)
                    payload["output"] = output.tolist()
                    self._send(200, payload)
                    return
                if path.endswith("/reset"):
                    pots = body.get("potentials")
                    inner.reset(
                        None if pots is None else np.asarray(pots, dtype=np.float64)
                    )
                    self._send(200, self._snapshot(inner))
                    return
        except (TypeError, ValueError) as exc:
            self._send(422, {"error": str(exc), "model": inner.model_id})
            return
        self._send(404, {"error": "not found"})


def make_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    registry: BioSimRouter | None = None,
) -> RouterHTTPServer:
    """Build a server that is not yet serving."""
    return RouterHTTPServer((host, port), registry if registry is not None else create_registry())


def base_url(server: RouterHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


@contextmanager
def running_router(
    *,
    host: str = DEFAULT_HOST,
    port: int = 0,
    registry: BioSimRouter | None = None,
) -> Iterator[tuple[str, RouterHTTPServer]]:
    """Start the router on a background thread (port 0 = ephemeral)."""
    server = make_server(host, port, registry)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield base_url(server), server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    registry: BioSimRouter | None = None,
) -> None:
    """Blocking serve loop for the CLI process."""
    server = make_server(host, port, registry)
    ids = ", ".join(server.registry.registered_ids())
    print(
        f"biorouter on {base_url(server)} (models: {ids}). "
        "Routes deployed bio-sim models; not OpenRouter-the-company, "
        "not the harness, not FlyWire dumps, not a marketplace.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down biorouter", flush=True)
    finally:
        server.shutdown()
        server.server_close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biorouter",
        description=(
            "OpenRouter routes existing LLMs; biorouter routes existing "
            "deployed biological simulation models. Same idea (model id), "
            "different substrate. Local stdlib HTTP process (POST /tick, "
            "POST /reset, GET /status). Lists flybrain.malecns when "
            "fly-harness[flybrain] and MaleCNS files are already on disk "
            "(otherwise omit / HTTP 404; never downloads). Not the harness, "
            "not FlyWire dumps, not a marketplace. Binds 127.0.0.1 by default."
        ),
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port (default: {DEFAULT_PORT})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
