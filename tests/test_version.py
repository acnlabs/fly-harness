"""Package version pin."""

from fly_harness import __version__


def test_package_version_is_0_5_3() -> None:
    assert __version__ == "0.5.3"
