"""JSON-serializable wrappers around FlyHarness.step — no MCP imports.

Call these from unit tests or from the FastMCP server. The default session is
the 24-neuron reflex fixture, not a whole-brain upload.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np

from fly_harness.harness import FlyHarness, StepResult

_POTENTIALS_LIST_LIMIT = 64


def default_reflex_harness() -> FlyHarness:
    """Fixture harness used when the MCP pack is started with no custom session."""
    from fly_harness.demo.codec import ReflexDecoder, ReflexEncoder
    from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome

    return FlyHarness(
        build_reflex_connectome(),
        ReflexEncoder(),
        ReflexDecoder(),
        decay=0.55,
        gain=0.45,
        sensory_indices=SENSORY_INDICES,
    )


def serialize_value(value: Any) -> Any:
    """Turn decoder output into JSON-friendly types."""
    as_env = getattr(value, "as_env_dict", None)
    if callable(as_env):
        raw = as_env()
        return {str(k): serialize_value(v) for k, v in dict(raw).items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {str(k): serialize_value(v) for k, v in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def serialize_step_result(result: StepResult, *, n_neurons: int) -> dict[str, Any]:
    potentials = np.asarray(result.potentials, dtype=np.float64).reshape(-1)
    payload: dict[str, Any] = {
        "action": serialize_value(result.action),
        "timestamp": float(result.timestamp),
        "n_neurons": int(n_neurons),
        "potentials_mean": float(np.mean(potentials)) if potentials.size else 0.0,
    }
    if potentials.size <= _POTENTIALS_LIST_LIMIT:
        payload["potentials"] = potentials.tolist()
    return payload


class HarnessSession:
    """Holds one FlyHarness and exposes step/reset/status as plain dicts."""

    def __init__(self, harness: FlyHarness | None = None) -> None:
        self.harness = harness if harness is not None else default_reflex_harness()

    def step(
        self,
        observation: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = dict(observation or {})
        for key, value in fields.items():
            if value is not None and key not in payload:
                payload[key] = value
        result = self.harness.step(payload)
        return serialize_step_result(result, n_neurons=self.harness.state.n_neurons)

    def reset(self) -> dict[str, Any]:
        self.harness.reset()
        return {
            "timestamp": float(self.harness.state.timestamp),
            "n_neurons": int(self.harness.state.n_neurons),
        }

    def status(self) -> dict[str, Any]:
        from fly_harness import __version__

        return {
            "version": __version__,
            "n_neurons": int(self.harness.state.n_neurons),
            "timestamp": float(self.harness.state.timestamp),
            "extension": "mcp",
            "kernel": "FlyHarness.step",
        }
