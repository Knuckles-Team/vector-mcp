"""Package initialization remains lightweight and explicit."""

import re

import vector_mcp


def test_package_exposes_semantic_version_without_import_side_effects() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", vector_mcp.__version__)
    assert vector_mcp.__all__ == ["__version__"]
