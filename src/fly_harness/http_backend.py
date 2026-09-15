"""Optional thin HTTP client for a remote ModelBackend or BioSimRouter.

Request bodies include a ``model`` field (OpenRouter-shaped). This is a
harness-side port so you can talk to a router *you* host later. It is not
a marketplace client, and this package does not run a public gateway.

Stdlib only (``urllib``). No extra dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

from fly_harness.backend import UnknownModelError


class HttpModelBackend:
    """POST ``/tick`` and ``/reset``; GET ``/status``. All carry ``model``."""

    def __init__(
        self,
        url: str,
        *,
        model_id: str,
        n_neurons: int | None = None,
        timeout: float = 5.0,
    ) -> None:
        if not url:
            raise ValueError("url must be a non-empty string")
        if not model_id:
            raise ValueError("model_id must be a non-empty string")
        if n_neurons is not None and n_neurons <= 0:
            raise ValueError("n_neurons must be positive")

        self.base_url = url.rstrip("/")
        self._model_id = model_id
        self.timeout = float(timeout)
        self._n_neurons = int(n_neurons) if n_neurons is not None else None
        self._timestamp = 0.0
        self.potentials = (
            np.zeros(self._n_neurons, dtype=np.float64)
            if self._n_neurons is not None
            else np.zeros(0, dtype=np.float64)
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def n_neurons(self) -> int:
        if self._n_neurons is None:
            self._refresh_status()
        assert self._n_neurons is not None
        return self._n_neurons

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def reset(self, potentials: np.ndarray | None = None) -> None:
        payload: dict[str, Any] = {
            "model": self._model_id,
            "potentials": None
            if potentials is None
            else np.asarray(potentials, dtype=np.float64).tolist(),
        }
        data = self._request("POST", "/reset", payload)
        self._apply_state(data, fallback_timestamp=0.0)

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        payload = {
            "model": self._model_id,
            "input": np.asarray(input_current, dtype=np.float64).tolist(),
        }
        data = self._request("POST", "/tick", payload)
        output = np.asarray(data["output"], dtype=np.float64)
        self._apply_state(data, output=output)
        return output

    def _refresh_status(self) -> None:
        query = urllib.parse.urlencode({"model": self._model_id})
        data = self._request("GET", f"/status?{query}", None)
        self._apply_state(data)

    def _apply_state(
        self,
        data: dict[str, Any],
        *,
        output: np.ndarray | None = None,
        fallback_timestamp: float | None = None,
    ) -> None:
        remote_model = data.get("model")
        if remote_model not in (None, self._model_id):
            raise UnknownModelError(str(remote_model))
        if output is not None:
            self.potentials = np.asarray(output, dtype=np.float64)
            self._n_neurons = int(data.get("n_neurons", output.shape[0]))
        elif "n_neurons" in data:
            self._n_neurons = int(data["n_neurons"])
            if self.potentials.shape != (self._n_neurons,):
                self.potentials = np.zeros(self._n_neurons, dtype=np.float64)
        if "output" in data and output is None:
            self.potentials = np.asarray(data["output"], dtype=np.float64)
        if "timestamp" in data:
            self._timestamp = float(data["timestamp"])
        elif fallback_timestamp is not None:
            self._timestamp = fallback_timestamp

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (400, 404):
                raise UnknownModelError(self._model_id) from exc
            raise RuntimeError(
                f"remote ModelBackend HTTP {exc.code} for model {self._model_id!r}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"remote ModelBackend unreachable at {self.base_url}: {exc}"
            ) from exc
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("remote ModelBackend returned a non-object JSON payload")
        return parsed
