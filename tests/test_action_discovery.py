"""Action-discovery standardization tests for vector-mcp tools.

Verifies that the action-routed MCP tools expose list_actions discovery and
raise a rich did-you-mean error on unknown actions, via the shared
agent_utilities.mcp.action_dispatch.resolve_action helper.
"""

import importlib
import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("LLM_BASE_URL", "http://test-url")


class _MockApiClient:
    def __init__(self):
        for name in (
            "create_collection",
            "add_documents",
            "delete_collection",
            "list_collections",
            "semantic_search",
            "lexical_search",
            "search",
        ):
            setattr(self, name, MagicMock(return_value={"ok": True}))


_mock_api_instance = _MockApiClient()
_mcp_server_module = importlib.import_module("vector_mcp.mcp_server")
get_mcp_instance = _mcp_server_module.get_mcp_instance


@pytest.fixture
def mock_client(monkeypatch):
    """Use a deterministic in-memory client at the MCP dependency boundary."""
    monkeypatch.setattr(_mcp_server_module, "get_client", lambda: _mock_api_instance)
    yield _mock_api_instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name,known_actions",
    [
        (
            "vector_collection_management",
            {
                "create_collection",
                "add_documents",
                "delete_collection",
                "list_collections",
            },
        ),
        (
            "vector_search",
            {"semantic_search", "lexical_search", "search"},
        ),
    ],
)
async def test_list_actions_returns_names(tool_name, known_actions, mock_client):
    """list_actions returns the discovery payload with all action names."""
    mcp, _, _ = get_mcp_instance()
    res = await mcp.call_tool(tool_name, {"action": "list_actions"})
    text = res.content[0].text
    assert "vector-mcp" in text
    for name in known_actions:
        assert name in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name",
    ["vector_collection_management", "vector_search"],
)
async def test_bogus_action_raises_did_you_mean(tool_name, mock_client):
    """An unknown action raises a ValueError mentioning list_actions."""
    mcp, _, _ = get_mcp_instance()
    with pytest.raises(Exception) as excinfo:
        await mcp.call_tool(tool_name, {"action": "definitely_not_real"})
    assert "list_actions" in str(excinfo.value)


@pytest.mark.asyncio
async def test_plural_alias_resolves(mock_client):
    """A plural alias resolves to the canonical singular action."""
    mock_client.search.reset_mock()
    mcp, _, _ = get_mcp_instance()
    await mcp.call_tool("vector_search", {"action": "searches", "question": "hello"})
    mock_client.search.assert_called_once()
