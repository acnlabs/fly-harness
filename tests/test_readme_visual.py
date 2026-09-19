"""README walking-fly visual is docs/usage display only; version stays 0.5.5."""

from __future__ import annotations

from pathlib import Path

from fly_harness import __version__

ROOT = Path(__file__).resolve().parents[1]
WALK_GIF = (
    "https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/overview_video.gif"
)


def test_package_version_stays_0_5_5() -> None:
    assert __version__ == "0.5.5"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.5.5"' in pyproject


def test_readme_leads_with_neuromechfly_walk_and_honest_copy() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    head = readme.split("## What this is not", 1)[0]
    assert WALK_GIF in head
    assert "docs/usage display only" in head
    assert "not a kernel feature" in head
    assert "24-neuron LIF fixture" in head
    assert "CPG" in head
    assert "FlyHarness.step" in head
    assert "above" in head.lower()
    assert "166,700" in head
    assert "consciousness" in head.lower()
    assert "google-hosted" in head.lower()
    assert "not better walking" in head.lower()
    assert "FlyGym's own controllers" in head
    assert "**v0.5.5**" in head
    assert head.index("NeuroMechFly") < head.index("Formula:")


def test_usage_walkthrough_shows_the_same_visual() -> None:
    usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
    assert WALK_GIF in usage
    assert "docs/usage display only" in usage
    assert "24-neuron LIF fixture" in usage
    assert "motor **CPG**" in usage or "motor CPG" in usage
    assert "not a kernel feature" in usage


def test_kernel_sources_do_not_reference_the_walk_gif() -> None:
    src = ROOT / "src" / "fly_harness"
    for name in ("__init__.py", "harness.py", "backend.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "overview_video.gif" not in text
        assert "NeuroMechFly walk" not in text
        assert WALK_GIF not in text
