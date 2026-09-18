"""Package version pin."""

from pathlib import Path

from fly_harness import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_is_0_5_5() -> None:
    assert __version__ == "0.5.5"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.5.5"' in pyproject
    assert 'Source = "https://github.com/acnlabs/fly-harness"' in pyproject
    assert (
        'Documentation = "https://github.com/acnlabs/fly-harness/blob/main/docs/docking.md"'
        in pyproject
    )
    assert 'Changelog = "https://github.com/acnlabs/fly-harness/releases"' in pyproject
