"""Package version pin."""

from fly_harness import __version__


def test_package_version_is_0_4_0() -> None:
    assert __version__ == "0.4.0"
