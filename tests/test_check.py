"""Load-check CLI: fixture LIF, one obs→action step, no extras."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fly_harness import __version__
from fly_harness.backend import DEFAULT_LIF_MODEL_ID
from fly_harness.check import LoadCheckReport, main, run_load_check
from fly_harness.demo.connectome import N_NEURONS


ROOT = Path(__file__).resolve().parents[1]


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    local_bin = str(Path.home() / ".local" / "bin")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (src, env.get("PYTHONPATH", "")) if part
    )
    env["PATH"] = os.pathsep.join(
        part for part in (local_bin, env.get("PATH", "")) if part
    )
    return env


def test_run_load_check_prints_contract_fields() -> None:
    report = run_load_check()
    assert report.model_id == DEFAULT_LIF_MODEL_ID
    assert report.n_neurons == N_NEURONS == 24
    assert report.timestamp == 1.0
    assert report.obs == {"touch_left": 1.0, "touch_right": 0.0}
    assert set(report.action) == {"turn", "forward", "brake"}
    assert report.action["turn"] in (-1, 0, 1)
    assert 0.0 <= float(report.action["forward"]) <= 1.0
    assert 0.0 <= float(report.action["brake"]) <= 1.0


def test_load_check_text_and_json_include_required_lines() -> None:
    report = run_load_check()
    text = report.format_text()
    for line in (
        f"model_id: {DEFAULT_LIF_MODEL_ID}",
        "n_neurons: 24",
        "timestamp: 1.0",
        "obs: touch_left=1.0 touch_right=0.0",
        "action:",
    ):
        assert line in text
    payload = json.loads(report.to_json())
    assert payload["model_id"] == DEFAULT_LIF_MODEL_ID
    assert payload["n_neurons"] == 24
    assert payload["timestamp"] == 1.0


def test_cli_stdout_default_and_json(capsys) -> None:
    assert main([]) == 0
    text = capsys.readouterr().out
    assert "model_id: fly-harness.in-process-lif" in text
    assert "n_neurons: 24" in text
    assert "timestamp:" in text
    assert "obs:" in text
    assert "action:" in text

    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_id"] == DEFAULT_LIF_MODEL_ID
    assert payload["n_neurons"] == 24
    assert "timestamp" in payload
    assert "obs" in payload
    assert "action" in payload


def test_module_entrypoints_without_extras() -> None:
    env = _child_env()
    for args in (
        [sys.executable, "-m", "fly_harness"],
        [sys.executable, "-m", "fly_harness.check"],
    ):
        completed = subprocess.run(
            args, check=True, capture_output=True, text=True, env=env
        )
        assert "model_id: fly-harness.in-process-lif" in completed.stdout
        assert "n_neurons: 24" in completed.stdout
        assert "timestamp:" in completed.stdout
        assert "obs:" in completed.stdout
        assert "action:" in completed.stdout
        assert completed.stderr == ""


def test_console_script_on_path_after_install() -> None:
    env = _child_env()
    script = shutil.which("fly-harness-check", path=env["PATH"])
    assert script is not None, "fly-harness-check should exist after pip install -e"
    completed = subprocess.run(
        [script], check=True, capture_output=True, text=True, env=env
    )
    assert "model_id: fly-harness.in-process-lif" in completed.stdout
    assert "n_neurons: 24" in completed.stdout
    assert "timestamp:" in completed.stdout
    assert "obs:" in completed.stdout
    assert "action:" in completed.stdout


def test_check_sources_do_not_bind_a_vendor_model() -> None:
    src = ROOT / "src" / "fly_harness"
    for name in ("check.py", "__main__.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "fly_harness.flybrain" not in text
        assert "fly_harness.c302" not in text
        assert "from flybrain" not in text
        assert "import flybrain" not in text
        assert "from c302" not in text
        assert "import c302" not in text
        assert "FakeC302" not in text
        assert "openworm" not in text.lower()


def test_usage_docs_cover_three_scenarios() -> None:
    assert __version__ == "0.5.4"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
    docking = (ROOT / "docs" / "docking.md").read_text(encoding="utf-8")
    for doc in (readme, usage):
        assert "fly-harness-check" in doc
        assert "Fixture reflex" in doc or "fixture reflex" in doc.lower()
        assert "ModelBackend" in doc
        assert "encode" in doc and "tick" in doc and "decode" in doc
        assert "flybrain.malecns" in doc
        assert "c302.celegans" in doc
        assert "usage, not kernel" in doc.lower() or "usage**, not kernel" in doc
        assert "not a chat" in doc.lower()
        assert "no extras" in doc.lower() or "No extras" in doc
        lower = doc.lower()
        assert "consciousness" in lower
        assert "not a consciousness" in lower
        assert "160k" in doc
        assert "not" in doc.lower() and "google-hosted" in lower
        assert "marketplace" not in lower or "not" in lower
    assert "docs/usage.md" in readme
    assert "docs/usage.md" in docking or "usage.md" in docking
    assert "fly-harness[flybrain]" in usage
    assert "not already a `ModelBackend`" in usage
    assert "fly-harness[c302]" in usage
    assert "There is **no**" in usage
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.5.4"' in pyproject
    assert "fly-harness-check" in pyproject
    assert LoadCheckReport is not None
