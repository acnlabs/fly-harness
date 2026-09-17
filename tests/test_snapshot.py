"""Optional ModelBackend snapshot/restore: toy LIF yes; missing backends error."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
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
    SnapshotUnsupportedError,
)
from fly_harness.c302.fake import FakeC302
from fly_harness.demo.codec import ReflexDecoder, ReflexEncoder, TouchObservation
from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome
from fly_harness.flybrain import FakeFlyBrain, FlyBrainBackend


class VectorEncoder:
    def encode(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float64)


class VectorDecoder:
    def decode(self, potentials: np.ndarray) -> np.ndarray:
        return np.asarray(potentials, dtype=np.float64)


class _TickOnlyBackend:
    """Duck-typed ModelBackend with no snapshot/restore mapping."""

    def __init__(self, n: int = 4) -> None:
        self._model_id = "tick.only"
        self._n_neurons = n
        self._timestamp = 0.0
        self.potentials = np.zeros(n, dtype=np.float64)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def reset(self, potentials: np.ndarray | None = None) -> None:
        self.potentials.fill(0.0 if potentials is None else 0.0)
        self._timestamp = 0.0

    def tick(self, input_current: np.ndarray) -> np.ndarray:
        self.potentials = np.asarray(input_current, dtype=np.float64)
        self._timestamp += 1.0
        return self.potentials.copy()


def _lif_harness() -> FlyHarness:
    return FlyHarness(
        build_reflex_connectome(),
        ReflexEncoder(),
        ReflexDecoder(),
        sensory_indices=SENSORY_INDICES,
        decay=0.2,
        gain=0.45,
    )


def test_toy_lif_snapshot_restore_continues_timestamp_and_potentials(tmp_path: Path) -> None:
    harness = _lif_harness()
    harness.step(TouchObservation(touch_left=1.0))
    harness.step(TouchObservation(touch_left=1.0))
    result = harness.step(TouchObservation(touch_left=1.0))
    assert result.timestamp == 3.0
    saved_pots = result.potentials.copy()
    path = tmp_path / "toy-lif.json"

    payload = harness.snapshot(path)
    assert path.is_file()
    assert payload["timestamp"] == 3.0
    assert json.loads(path.read_text(encoding="utf-8"))["timestamp"] == 3.0

    later = _lif_harness()
    assert later.backend.timestamp == 0.0
    np.testing.assert_allclose(later.state.potentials, 0.0)

    later.restore(path)
    assert later.backend.timestamp == 3.0
    np.testing.assert_allclose(later.state.potentials, saved_pots)

    continued = later.step(TouchObservation(touch_left=1.0))
    assert continued.timestamp == 4.0
    assert not np.allclose(continued.potentials, 0.0)


def test_toy_lif_snapshot_restore_new_process(tmp_path: Path) -> None:
    snap = tmp_path / "toy-lif.json"
    expected = tmp_path / "expected.json"
    script = tmp_path / "save_snap.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "from fly_harness.backend import InProcessLifBackend",
                "from fly_harness.demo.codec import ReflexEncoder, TouchObservation",
                "from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome",
                f"snap = Path({str(snap)!r})",
                f"expected = Path({str(expected)!r})",
                "backend = InProcessLifBackend(",
                "    build_reflex_connectome(), sensory_indices=SENSORY_INDICES, decay=0.2, gain=0.45",
                ")",
                "enc = ReflexEncoder()",
                "for _ in range(3):",
                "    backend.tick(enc.encode(TouchObservation(touch_left=1.0)))",
                "expected.write_text(json.dumps({",
                '    "timestamp": float(backend.timestamp),',
                '    "potentials": backend.potentials.tolist(),',
                "}))",
                "backend.snapshot(snap)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(script)], check=True)

    marker = json.loads(expected.read_text(encoding="utf-8"))
    fresh = InProcessLifBackend(
        build_reflex_connectome(),
        sensory_indices=SENSORY_INDICES,
        decay=0.2,
        gain=0.45,
    )
    assert fresh.timestamp == 0.0
    fresh.restore(snap)
    assert fresh.timestamp == marker["timestamp"] == 3.0
    np.testing.assert_allclose(fresh.potentials, np.asarray(marker["potentials"], dtype=np.float64))

    next_pots = fresh.tick(ReflexEncoder().encode(TouchObservation(touch_left=1.0)))
    assert fresh.timestamp == 4.0
    assert not np.allclose(next_pots, 0.0)


def test_direct_and_router_forward_lif_snapshot(tmp_path: Path) -> None:
    inner = InProcessLifBackend(
        build_reflex_connectome(),
        sensory_indices=SENSORY_INDICES,
        decay=0.2,
        gain=0.45,
    )
    wrapped = DirectBioSimBackend(inner)
    wrapped.tick(ReflexEncoder().encode(TouchObservation(touch_left=1.0)))
    path = tmp_path / "direct-lif.json"
    wrapped.snapshot(path)

    router = BioSimRouter({"fly-harness.in-process-lif": wrapped}, model_id="fly-harness.in-process-lif")
    later = InProcessLifBackend(build_reflex_connectome(), sensory_indices=SENSORY_INDICES)
    later.restore(path)
    assert later.timestamp == 1.0
    router.restore(path)
    assert router.timestamp == 1.0


def test_unsupported_backends_raise_clearly() -> None:
    fake = FakeDeployedSim(4, model_id="fake.deployed")
    harness = FlyHarness(encoder=VectorEncoder(), decoder=VectorDecoder(), backend=fake)
    with pytest.raises(SnapshotUnsupportedError, match="not a silent no-op"):
        harness.snapshot("unused.json")
    with pytest.raises(SnapshotUnsupportedError, match="FakeDeployedSim"):
        harness.restore("unused.json")

    direct = DirectBioSimBackend(n_neurons=3, model_id="vendor.fake")
    with pytest.raises(SnapshotUnsupportedError, match="no snapshot"):
        direct.snapshot()

    http = HttpModelBackend("http://127.0.0.1:9", model_id="remote.sim", n_neurons=2)
    with pytest.raises(SnapshotUnsupportedError, match="does not invent a checkpoint wire format"):
        http.snapshot()
    with pytest.raises(SnapshotUnsupportedError, match="does not invent a checkpoint wire format"):
        http.restore({})

    missing = _TickOnlyBackend()
    loop = FlyHarness(encoder=VectorEncoder(), decoder=VectorDecoder(), backend=missing)
    with pytest.raises(SnapshotUnsupportedError, match="optional ModelBackend port"):
        loop.snapshot()
    with pytest.raises(SnapshotUnsupportedError, match="restore"):
        loop.restore({})

    router = BioSimRouter({"fake.deployed": fake}, model_id="fake.deployed")
    with pytest.raises(SnapshotUnsupportedError, match="FakeDeployedSim"):
        router.snapshot()


def test_flybrain_and_c302_fixtures_have_no_invented_checkpoint() -> None:
    fly = FlyBrainBackend(FakeFlyBrain())
    worm = FakeC302()
    for backend in (fly, worm):
        loop = FlyHarness(encoder=VectorEncoder(), decoder=VectorDecoder(), backend=backend)
        with pytest.raises(SnapshotUnsupportedError, match="does not invent vendor checkpoint formats"):
            loop.snapshot()
        with pytest.raises(SnapshotUnsupportedError, match="no restore"):
            loop.restore({})
