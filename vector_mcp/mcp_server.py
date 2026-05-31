#!/usr/bin/python
import warnings

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from fastmcp.utilities.logging import get_logger
from pydantic import Field

# Filter RequestsDependencyWarning early to prevent log spam
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")

import logging
import os
import sys
from pathlib import Path
from typing import Any

from agent_utilities.base_utilities import to_boolean
from agent_utilities.mcp_utilities import create_mcp_server
from dotenv import find_dotenv, load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse

from vector_mcp.auth import get_client

__version__ = "1.32.1"

logger = get_logger(name="vector-mcp")
logger.setLevel(logging.INFO)


def register_collection_management_tools(mcp: FastMCP):
    @mcp.tool(tags={"collection_management"})
    async def vector_collection_management(
        action: str = Field(
            description="Action to perform. Must be one of: 'create_collection', 'add_documents', 'delete_collection', 'list_collections'"
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
        overwrite: bool | None = Field(default=None, description="overwrite"),
        document_directory: Path | str | None = Field(
            default=None, description="document directory"
        ),
        document_paths: Path | str | None = Field(
            default=None, description="document paths"
        ),
        document_contents: list[str] | None = Field(
            default=None, description="document contents"
        ),
        confirm: bool | None = Field(default=None, description="confirm"),
        client=Depends(get_client),
    ) -> dict:
        """Manage collection management operations.

        Actions:
          - 'create_collection': Creates a new collection or retrieves an existing one in the vector database.
          - 'add_documents': Adds documents to an existing collection in the vector database.
          - 'delete_collection': Deletes a collection from the vector database.
          - 'list_collections': Lists all collections in the vector database.
        """
        kwargs: dict[str, Any]
        if action == "create_collection":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
                "collection_name": collection_name,
                "overwrite": overwrite,
                "document_directory": document_directory,
                "document_paths": document_paths,
                "document_contents": document_contents,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.create_collection(**kwargs)
        if action == "add_documents":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
                "collection_name": collection_name,
                "document_directory": document_directory,
                "document_paths": document_paths,
                "document_contents": document_contents,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.add_documents(**kwargs)
        if action == "delete_collection":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
                "collection_name": collection_name,
                "confirm": confirm,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.delete_collection(**kwargs)
        if action == "list_collections":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.list_collections(**kwargs)
        raise ValueError(
            f"Unknown action: {action}. Must be one of: create_collection', 'add_documents', 'delete_collection', 'list_collections"
        )


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
            return client.semantic_search(**kwargs)
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
            return client.lexical_search(**kwargs)
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
            return client.search(**kwargs)
        raise ValueError(
            f"Unknown action: {action}. Must be one of: semantic_search', 'lexical_search', 'search"
        )


def get_mcp_instance() -> tuple[Any, ...]:
    """Initialize and return the MCP instance."""
    load_dotenv(find_dotenv())
    args, mcp, middlewares = create_mcp_server(
        name="vector-mcp MCP",
        version=__version__,
        instructions="vector-mcp MCP Server — Condensed Action-Routed Tools.",
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})

    DEFAULT_COLLECTION_MANAGEMENTTOOL = to_boolean(
        os.getenv("COLLECTION_MANAGEMENTTOOL", "True")
    )
    if DEFAULT_COLLECTION_MANAGEMENTTOOL:
        register_collection_management_tools(mcp)
    DEFAULT_SEARCHTOOL = to_boolean(os.getenv("SEARCHTOOL", "True"))
    if DEFAULT_SEARCHTOOL:
        register_search_tools(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)
    return mcp, args, middlewares


def mcp_server() -> None:
    mcp, args, middlewares = get_mcp_instance()
    print(f"vector-mcp MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


if __name__ == "__main__":
    mcp_server()
