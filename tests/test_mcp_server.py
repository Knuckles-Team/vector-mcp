import importlib
import json
from unittest.mock import MagicMock, patch

import pytest


class MockApiClient:
    def __init__(self):
        self.create_collection = MagicMock()
        self.add_documents = MagicMock()
        self.delete_collection = MagicMock()
        self.list_collections = MagicMock()
        self.semantic_search = MagicMock()
        self.lexical_search = MagicMock()
        self.search = MagicMock()

    def reset_mock(self):
        self.create_collection.reset_mock()
        self.add_documents.reset_mock()
        self.delete_collection.reset_mock()
        self.list_collections.reset_mock()
        self.semantic_search.reset_mock()
        self.lexical_search.reset_mock()
        self.search.reset_mock()


mock_api_instance = MockApiClient()

mcp_server_module = importlib.import_module("vector_mcp.mcp_server")
get_mcp_instance = mcp_server_module.get_mcp_instance
mcp_server = mcp_server_module.mcp_server


@pytest.fixture
def mock_client_methods(monkeypatch):
    """Reset the mock client methods."""
    mock_api_instance.reset_mock()
    monkeypatch.setattr(mcp_server_module, "get_client", lambda: mock_api_instance)
    yield mock_api_instance


@pytest.mark.asyncio
async def test_health_check_route():
    """Assert the /health custom endpoint returns OK."""
    mcp, _, _ = get_mcp_instance()
    route = next(
        route for route in mcp._additional_http_routes if route.path == "/health"
    )
    response = await route.endpoint(None)
    assert response.status_code == 200
    assert json.loads(response.body.decode()) == {"status": "ok"}


@pytest.mark.asyncio
async def test_vector_collection_management_tools(mock_client_methods):
    """Verify tool registration and dependency execution for collection management."""
    mcp, _, _ = get_mcp_instance()

    # 1. Test create_collection
    mock_client_methods.create_collection.return_value = {"status": "created"}
    res = await mcp.call_tool(
        "vector_collection_management",
        {
            "action": "create_collection",
            "db_type": "epistemic_graph",
            "collection_name": "test_collection",
            "overwrite": True,
        },
    )
    # FastMCP call_tool returns a ToolResult. Inspect res.content[0].text
    assert "status" in res.content[0].text
    assert "created" in res.content[0].text
    mock_client_methods.create_collection.assert_called_once_with(
        db_type="epistemic_graph", collection_name="test_collection", overwrite=True
    )

    # 2. Test add_documents
    mock_client_methods.add_documents.return_value = {"status": "added"}
    mock_client_methods.reset_mock()
    res = await mcp.call_tool(
        "vector_collection_management",
        {
            "action": "add_documents",
            "db_type": "epistemic_graph",
            "collection_name": "test_collection",
            "document_contents": ["Hello world"],
        },
    )
    assert "status" in res.content[0].text
    assert "added" in res.content[0].text
    mock_client_methods.add_documents.assert_called_once_with(
        db_type="epistemic_graph",
        collection_name="test_collection",
        document_contents=["Hello world"],
    )

    # 3. Test delete_collection
    mock_client_methods.delete_collection.return_value = {"status": "deleted"}
    mock_client_methods.reset_mock()
    res = await mcp.call_tool(
        "vector_collection_management",
        {
            "action": "delete_collection",
            "db_type": "epistemic_graph",
            "collection_name": "test_collection",
            "confirm": True,
        },
    )
    assert "status" in res.content[0].text
    assert "deleted" in res.content[0].text
    mock_client_methods.delete_collection.assert_called_once_with(
        db_type="epistemic_graph", collection_name="test_collection", confirm=True
    )

    # 4. Test list_collections
    mock_client_methods.list_collections.return_value = {
        "collections": ["col1", "col2"]
    }
    mock_client_methods.reset_mock()
    res = await mcp.call_tool(
        "vector_collection_management",
        {
            "action": "list_collections",
            "db_type": "epistemic_graph",
        },
    )
    assert "col1" in res.content[0].text
    mock_client_methods.list_collections.assert_called_once_with(
        db_type="epistemic_graph"
    )

    # 5. Test unknown action
    with pytest.raises(Exception, match="Unknown action 'invalid_action'"):
        await mcp.call_tool(
            "vector_collection_management", {"action": "invalid_action"}
        )


@pytest.mark.asyncio
async def test_vector_search_tools(mock_client_methods):
    """Verify tool registration and dependency execution for vector search."""
    mcp, _, _ = get_mcp_instance()

    # 1. Test semantic_search
    mock_client_methods.semantic_search.return_value = {"results": []}
    res = await mcp.call_tool(
        "vector_search",
        {
            "action": "semantic_search",
            "db_type": "epistemic_graph",
            "collection_name": "test_collection",
            "question": "What is AI?",
            "number_results": 5,
        },
    )
    assert "results" in res.content[0].text
    mock_client_methods.semantic_search.assert_called_once_with(
        db_type="epistemic_graph",
        collection_name="test_collection",
        question="What is AI?",
        number_results=5,
    )

    # 2. Test lexical_search
    mock_client_methods.lexical_search.return_value = {"results": []}
    mock_client_methods.reset_mock()
    res = await mcp.call_tool(
        "vector_search",
        {
            "action": "lexical_search",
            "db_type": "epistemic_graph",
            "collection_name": "test_collection",
            "question": "What is AI?",
            "number_results": 3,
        },
    )
    assert "results" in res.content[0].text
    mock_client_methods.lexical_search.assert_called_once_with(
        db_type="epistemic_graph",
        collection_name="test_collection",
        question="What is AI?",
        number_results=3,
    )

    # 3. Test hybrid search
    mock_client_methods.search.return_value = {"results": []}
    mock_client_methods.reset_mock()
    res = await mcp.call_tool(
        "vector_search",
        {
            "action": "search",
            "db_type": "epistemic_graph",
            "collection_name": "test_collection",
            "question": "What is AI?",
            "number_results": 2,
            "semantic_weight": 0.7,
            "lexical_weight": 0.3,
            "rrf_k": 60,
        },
    )
    assert "results" in res.content[0].text
    mock_client_methods.search.assert_called_once_with(
        db_type="epistemic_graph",
        collection_name="test_collection",
        question="What is AI?",
        number_results=2,
        semantic_weight=0.7,
        lexical_weight=0.3,
        rrf_k=60,
    )

    # 4. Test unknown action
    with pytest.raises(Exception, match="Unknown action 'invalid_action'"):
        await mcp.call_tool("vector_search", {"action": "invalid_action"})


def test_mcp_server_cli_execution():
    """Verify that mcp_server CLI runs as expected with mocked transport runs."""
    # Test stdio transport
    with (
        patch("sys.argv", ["mcp_server", "--transport", "stdio"]),
        patch("fastmcp.FastMCP.run") as mock_run,
    ):
        mcp_server()
        mock_run.assert_called_once_with(transport="stdio")

    # Test sse transport
    with (
        patch(
            "sys.argv",
            [
                "mcp_server",
                "--transport",
                "sse",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
        ),
        patch("fastmcp.FastMCP.run") as mock_run,
    ):
        mcp_server()
        mock_run.assert_called_once_with(transport="sse", host="127.0.0.1", port=8000)

    # Test streamable-http transport
    with (
        patch(
            "sys.argv",
            [
                "mcp_server",
                "--transport",
                "streamable-http",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
        ),
        patch("fastmcp.FastMCP.run") as mock_run,
    ):
        mcp_server()
        mock_run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8000
        )

    # Test invalid transport (which results in SystemExit 2 via argparse parsing)
    with (
        patch("sys.argv", ["mcp_server", "--transport", "invalid-transport"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        mcp_server()
    assert exc_info.value.code == 2
