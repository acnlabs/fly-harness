"""Package version pin."""

from fly_harness import __version__


def test_package_version_is_0_5_1() -> None:
    assert __version__ == "0.5.1"
