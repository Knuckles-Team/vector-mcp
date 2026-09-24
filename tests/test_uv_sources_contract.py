"""Tests for the consumer's canonical local uv source declarations."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_local_uv_sources_are_direct_workspace_siblings() -> None:
    manifest = Path(__file__).parents[1] / "pyproject.toml"
    with manifest.open("rb") as handle:
        sources = tomllib.load(handle)["tool"]["uv"]["sources"]

    expected = {
        "agent-connector-sdk",
        "agent-utilities",
        "epistemic-graph",
        "langfuse-agent",
    }
    assert set(sources) == expected
    for name in expected:
        assert sources[name]["path"] == f".uv-workspace-siblings/{name}"
        assert sources[name]["editable"] is True
