"""MCP tools for search operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from typing import Any

from agent_utilities.mcp_utilities import resolve_action, run_blocking
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from vector_mcp.auth import get_client


def register_search_tools(mcp: FastMCP):
    @mcp.tool(tags={"search"})
    async def vector_search(
        action: str = Field(
            description="Action to perform. Must be one of: 'semantic_search', 'lexical_search', 'search'"
        ),
        db_type: str | None = Field(default=None, description="db type"),
        db_path: str | None = Field(default=None, description="db path"),
        host: str | None = Field(default=None, description="host"),
        port: str | None = Field(default=None, description="port"),
        db_name: str | None = Field(default=None, description="db name"),
        username: str | None = Field(default=None, description="username"),
        password: str | None = Field(default=None, description="password"),
        collection_name: str | None = Field(
            default=None, description="collection name"
        ),
        question: str | None = Field(default=None, description="question"),
        number_results: int | None = Field(default=None, description="number results"),
        semantic_weight: float | None = Field(
            default=None, description="semantic weight"
        ),
        bm25_weight: float | None = Field(default=None, description="bm25 weight"),
        rrf_k: int | None = Field(default=None, description="rrf k"),
        client=Depends(get_client),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """Manage search operations.

        Actions:
          - 'semantic_search': Retrieves and gathers related knowledge from the vector database instance using the question variable.
          - 'lexical_search': This is a lexical or term based search that retrieves and gathers related knowledge from the database instance using the question variable via BM25.
          - 'search': Performs a hybrid search combining semantic (vector) and lexical (BM25) methods.
        """
        if ctx:
            try:
                await ctx.info("Executing tool...")
            except Exception:
                pass
        resolved = resolve_action(
            action,
            ("semantic_search", "lexical_search", "search"),
            service="vector-mcp",
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        kwargs: dict[str, Any]
        if action == "semantic_search":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
                "collection_name": collection_name,
                "question": question,
                "number_results": number_results,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return await run_blocking(client.semantic_search, **kwargs)
        if action == "lexical_search":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
                "collection_name": collection_name,
                "question": question,
                "number_results": number_results,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return await run_blocking(client.lexical_search, **kwargs)
        if action == "search":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
                "collection_name": collection_name,
                "question": question,
                "number_results": number_results,
                "semantic_weight": semantic_weight,
                "bm25_weight": bm25_weight,
                "rrf_k": rrf_k,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return await run_blocking(client.search, **kwargs)
        raise ValueError(
            f"Unknown action: {action}. Must be one of: semantic_search', 'lexical_search', 'search"
        )
