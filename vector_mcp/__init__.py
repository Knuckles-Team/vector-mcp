"""Governed vector retrieval provider package."""

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path


def _source_version() -> str:
    """Read the project version when importing directly from an unpackaged checkout."""
    try:
        project = Path(__file__).resolve().parents[1] / "pyproject.toml"
        return str(
            tomllib.loads(project.read_text(encoding="utf-8"))["project"]["version"]
        )
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return "0.0.0"


try:
    __version__ = _distribution_version("vector-mcp")
except PackageNotFoundError:
    __version__ = _source_version()

__all__ = ["__version__"]
