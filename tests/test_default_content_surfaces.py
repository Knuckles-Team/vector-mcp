"""CONN-VECTOR-MCP-CONNECTOR-CONTENT-R001: the server lists its packaged skills, ontology and prompts by default."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest
from fastmcp import Client

PACKAGE = "vector_mcp"
ROOT = Path(list(importlib.import_module(PACKAGE).__path__)[0])

pytestmark = pytest.mark.spec("CONN-VECTOR-MCP-CONNECTOR-CONTENT-R001")


def _server():
    from agent_connector_sdk.connector import build_server

    return build_server(PACKAGE, command_args=[])[1]


async def _listings():
    async with Client(_server()) as client:
        resources = {str(r.uri) for r in await client.list_resources()}
        prompts = {p.name for p in await client.list_prompts()}
    return resources, prompts


def test_default_content_surfaces() -> None:
    resources, prompts = asyncio.run(_listings())
    if list(ROOT.glob("skills/*/SKILL.md")):
        assert any(u.startswith("skill://") for u in resources)
    if list(ROOT.glob("ontology/*.ttl")):
        assert any(u.startswith("ontology://vector-mcp/") for u in resources)
    for path in ROOT.glob("prompts/*.json"):
        if not path.name.startswith("_"):
            assert path.stem in prompts
