"""Docking guide is docs-only: core stays model-agnostic; version pin follows the package."""

from __future__ import annotations

from pathlib import Path

from fly_harness import __version__


ROOT = Path(__file__).resolve().parents[1]


def test_package_version_stays_0_5_5() -> None:
    assert __version__ == "0.5.5"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.5.5"' in pyproject


def test_kernel_sources_do_not_import_flybrain_or_flygym() -> None:
    src = ROOT / "src" / "fly_harness"
    for name in ("__init__.py", "harness.py", "backend.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "fly_harness.flybrain" not in text
        assert "fly_harness.flygym" not in text
        assert "fly_harness.c302" not in text
        assert "from flybrain" not in text
        assert "import flybrain" not in text
        assert "from c302" not in text
        assert "import c302" not in text
        assert "openworm" not in text.lower()
        assert "FakeC302" not in text
        assert "docs/docking" not in text


def test_readme_points_at_docking_guide() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/docking.md" in readme
    assert "docs/usage.md" in readme
    assert "Docking your Model" in readme
    assert "Usage walkthrough" in readme
    assert "fly-harness-check" in readme
    assert "one listed" in readme.lower()
    assert "the harness docks flybrain" in readme  # negated in the same sentence
    assert "not “the harness docks flybrain”" in readme or 'not "the harness docks flybrain"' in readme
    assert "c302.celegans" in readme
    assert "FakeC302" in readme
    assert "hermaphrodite" in readme.lower()


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
        "snapshot()",
        "restore(...)",
        "SnapshotUnsupportedError",
        "not a silent no-op",
        "BrainState.save",
        "BrainState.load",
        "flybrain / c302 / NEURON",
        "fly-harness-check --save",
        "biorouter",
        "FakeDeployedSim",
        "UnknownModelError",
        "flybrain.malecns",
        "one listed",
        "examples/biorouter_loop.py",
        "c302.celegans",
        "FakeC302",
        "302",
        "hermaphrodite",
        "usage.md",
        "fly-harness-check",
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
    assert "openworm" in lower
    assert "hosted" in lower
    assert "not a fly-harness extra" in lower or "not a harness extra" in lower
    assert "deploy a sim" in lower or "list an id" in lower


def test_no_resume_product_surface() -> None:
    """Checkpoint is a ModelBackend port, not a fourth how-to or check --save."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
    check = (ROOT / "src" / "fly_harness" / "check.py").read_text(encoding="utf-8")
    assert "fly-harness-check --save" not in readme
    assert "fly-harness-check --save" not in usage
    assert "--save" not in check
    assert "\n## 4." not in usage
    assert "\n## 1." in usage
    assert "\n## 2." in usage
    assert "\n## 3." in usage
    assert "Usage walkthrough" in readme
    # README still has three numbered how-tos, not a resume scenario
    assert "### 1. Fixture reflex" in readme
    assert "### 2. Direct" in readme
    assert "### 3. biorouter" in readme
    assert "### 4." not in readme
