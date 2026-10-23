"""Vector-store clients come from the SDK extras, not from connector pins."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

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
        "postgres": "agent-connector-sdk[vector]>=0.4.1,<1.0.0",
        "qdrant": "agent-connector-sdk[vector]>=0.4.1,<1.0.0",
        "mongodb": "agent-connector-sdk[vector]>=0.4.1,<1.0.0",
        "chromadb": "agent-connector-sdk[vector-chroma]>=0.4.1,<1.0.0",
    }
    for extra, requirement in expected.items():
        assert extras[extra] == [requirement]
    pinned = [*project["dependencies"], *(r for reqs in extras.values() for r in reqs)]
    for requirement in pinned:
        name = requirement.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0]
        assert name.strip().lower() not in STORE_CLIENTS, requirement
