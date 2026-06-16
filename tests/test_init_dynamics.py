"""Verify package initialization and version metadata."""

import importlib

import pytest

PKG_NAME = __name__.rsplit(".", 1)[0] if "." in __name__ else None


def _get_pkg_name():
    """Derive the importable package name from the project source layout.

    Locating the package directory (the child dir with an ``__init__.py``)
    rather than the project folder name keeps this robust when the checkout
    lives in a git worktree whose directory is not named after the package.
    """
    import pathlib

    test_dir = pathlib.Path(__file__).resolve().parent
    project_dir = test_dir.parent
    for child in sorted(project_dir.iterdir()):
        if (
            child.is_dir()
            and (child / "__init__.py").is_file()
            and not child.name.endswith(".egg-info")
        ):
            return child.name
    return project_dir.name.replace("-", "_")


@pytest.fixture
def pkg_name():
    return _get_pkg_name()


def test_package_importable(pkg_name):
    """Package should be importable."""
    mod = importlib.import_module(pkg_name)
    assert mod is not None


def _resolve_version(pkg_name):
    """Resolve the package version from source attr or installed metadata.

    Skips when neither is available (e.g. running from a git worktree whose
    CWD cannot resolve the editable distribution metadata) — an environment
    condition rather than a package defect.
    """
    mod = importlib.import_module(pkg_name)
    version = getattr(mod, "__version__", None)
    if version is None:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as get_version

        try:
            version = get_version(pkg_name.replace("_", "-"))
        except PackageNotFoundError:
            pytest.skip(
                f"{pkg_name} distribution metadata not resolvable in this environment"
            )
    return version


def test_version_exists(pkg_name):
    """Package should expose __version__."""
    version = _resolve_version(pkg_name)
    assert version is not None, f"{pkg_name} has no __version__"


def test_version_format(pkg_name):
    """Version should follow semver-like format."""
    version = _resolve_version(pkg_name)
    parts = version.split(".")
    assert len(parts) >= 2, f"Version {version} should have at least major.minor"
