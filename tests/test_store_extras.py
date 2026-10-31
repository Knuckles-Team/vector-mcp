"""Vector-store clients come from the SDK extras, not from connector pins."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
STORE_CLIENTS = (
    "qdrant-client",
    "psycopg",
    "pymongo",
    "chromadb",
    "llama-index-vector-stores-chroma",
)


@pytest.mark.spec("VECTOR-R001.7")
def test_store_extras_come_from_the_sdk() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    expected = {
        "postgres": "vector",
        "qdrant": "vector",
        "mongodb": "vector",
        "chromadb": "vector-chroma",
    }
    for extra, sdk_extra in expected.items():
        assert len(extras[extra]) == 1
        sdk = Requirement(extras[extra][0])
        assert sdk.name == "scion-sdk"
        assert sdk_extra in sdk.extras
        floor = max(Version(s.version) for s in sdk.specifier if s.operator == ">=")
        assert floor >= Version("0.6.3")
        assert not sdk.specifier.contains("0.7.0")
    pinned = [*project["dependencies"], *(r for reqs in extras.values() for r in reqs)]
    for requirement in pinned:
        name = requirement.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0]
        assert name.strip().lower() not in STORE_CLIENTS, requirement
