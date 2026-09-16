"""Docking guide is docs-only: core stays model-agnostic; version pin follows the package."""

from __future__ import annotations

from pathlib import Path

from fly_harness import __version__


ROOT = Path(__file__).resolve().parents[1]


def test_package_version_stays_0_5_1() -> None:
    assert __version__ == "0.5.1"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.5.1"' in pyproject


def test_kernel_sources_do_not_import_flybrain_or_flygym() -> None:
    src = ROOT / "src" / "fly_harness"
    for name in ("__init__.py", "harness.py", "backend.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "fly_harness.flybrain" not in text
        assert "fly_harness.flygym" not in text
        assert "from flybrain" not in text
        assert "import flybrain" not in text
        assert "docs/docking" not in text


def test_readme_points_at_docking_guide() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/docking.md" in readme
    assert "Docking your Model" in readme
    assert "one listed" in readme.lower()
    assert "the harness docks flybrain" in readme  # negated in the same sentence
    assert "not “the harness docks flybrain”" in readme or 'not "the harness docks flybrain"' in readme


def test_docking_guide_covers_contract_and_listing() -> None:
    guide = (ROOT / "docs" / "docking.md").read_text(encoding="utf-8")
    for needle in (
        "ModelBackend",
        "tick",
        "reset",
        "n_neurons",
        "model_id",
        "timestamp",
        "FlyHarness.step",
        "biorouter",
        "FakeDeployedSim",
        "UnknownModelError",
        "flybrain.malecns",
        "one listed",
        "examples/biorouter_loop.py",
    ):
        assert needle in guide, needle
    assert "the harness docks flybrain" in guide  # only as a negation
    assert "not “the harness docks flybrain”" in guide or 'not "the harness docks flybrain"' in guide
    lower = guide.lower()
    assert "consciousness" in lower
    assert "not a consciousness" in lower
    assert "160k" in guide
    assert "not ~160k" in guide or "not 160k" in guide
    assert "will not download" in lower or "not download" in lower
    assert "marketplace" in lower
