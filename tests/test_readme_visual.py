"""README walking-fly visual is our clip, not FlyGym's GIF; version stays 0.5.5."""

from __future__ import annotations

from pathlib import Path

from fly_harness import __version__

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media" / "visual-demo"
FLYGYM_GIF = (
    "https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/overview_video.gif"
)
CLIP = "docs/media/visual-demo/fly-walking.mp4"


def test_package_version_stays_0_5_5() -> None:
    assert __version__ == "0.5.5"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.5.5"' in pyproject


def test_visual_demo_files_are_committed() -> None:
    mp4 = MEDIA / "fly-walking.mp4"
    walking = MEDIA / "01-walking.png"
    left = MEDIA / "02-touch-left.png"
    right = MEDIA / "03-touch-right.png"
    assert mp4.is_file()
    assert walking.is_file()
    assert left.is_file()
    assert right.is_file()
    data = mp4.read_bytes()
    assert data[4:8] == b"ftyp"
    assert len(data) > 1_000_000
    assert walking.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert left.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert right.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_readme_leads_with_our_clip_not_flygym_gif() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    head = readme.split("## Install", 1)[0]
    assert CLIP in head
    assert "docs/media/visual-demo/01-walking.png" in head
    assert FLYGYM_GIF not in readme
    assert "overview_video.gif" not in readme
    assert "NeLy-EPFL/_media" not in readme
    assert "not a kernel feature" in head
    assert "24-neuron LIF fixture" in head
    assert "CPG" in head
    assert "FlyHarness.step" in head
    assert "**v0.5.5**" in head
    first_bash = readme.index("```bash")
    assert readme.index(CLIP) < first_bash
    assert "## Usage walkthrough" not in readme
    assert "### 1. Fixture reflex" not in readme


def test_usage_walkthrough_shows_the_same_visual() -> None:
    usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
    assert "media/visual-demo/fly-walking.mp4" in usage
    assert FLYGYM_GIF not in usage
    assert "overview_video.gif" not in usage
    assert "docs/usage display only" in usage
    assert "24-neuron LIF fixture" in usage
    assert "motor **CPG**" in usage or "motor CPG" in usage
    assert "not a kernel feature" in usage


def test_kernel_sources_do_not_reference_the_walk_clip() -> None:
    src = ROOT / "src" / "fly_harness"
    for name in ("__init__.py", "harness.py", "backend.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "overview_video.gif" not in text
        assert "fly-walking.mp4" not in text
        assert "NeuroMechFly walk" not in text
        assert FLYGYM_GIF not in text
