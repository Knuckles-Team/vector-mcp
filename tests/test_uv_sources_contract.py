"""Dependencies resolve from PyPI only: no local or git uv source overrides."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_dependencies_have_no_local_or_git_uv_sources() -> None:
    manifest = Path(__file__).parents[1] / "pyproject.toml"
    with manifest.open("rb") as handle:
        sources = tomllib.load(handle).get("tool", {}).get("uv", {}).get("sources", {})

    overridden = {
        name: spec
        for name, spec in sources.items()
        if isinstance(spec, dict) and ({"path", "git"} & set(spec))
    }
    assert overridden == {}
