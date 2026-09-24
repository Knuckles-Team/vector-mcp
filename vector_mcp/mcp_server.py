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
import re
import sys
from dataclasses import dataclass
from typing import Any, Literal

from agent_connector_sdk.config import load_config, setting
from agent_connector_sdk.mcp.action_dispatch import resolve_action
from agent_connector_sdk.mcp.concurrency import run_blocking
from agent_connector_sdk.mcp.server import create_mcp_server
from agent_connector_sdk.mcp.tool_surface import register_tool_surface

from vector_mcp import __version__
from vector_mcp.backend_policy import ensure_backend_available
from vector_mcp.document_inputs import resolve_document_inputs
from vector_mcp.vector_api import get_client

logger = get_logger(name="vector-mcp")
logger.setLevel(logging.INFO)

_COLLECTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")


def _backend(value: str | None) -> str:
    configured = value or str(
        setting("DATABASE_TYPE", "epistemic_graph") or "epistemic_graph"
    )
    return ensure_backend_available(configured)


def _collection(value: str | None) -> str | None:
    if value is None:
        return None
    if not _COLLECTION_NAME.fullmatch(value):
        raise ValueError("Collection name is invalid")
    return value


def _entitled(namespace: str, names: list[str]) -> list[str]:
    """Filter ``names`` to what the calling identity may reach.

    Routes the set through agent-utilities' shared identity-scoped resolver
    (CONCEPT:AU-OS.identity.identity-scoped-resource-autoload): a caller's
    Okta/Keycloak groups decide which resources they see by default. The
    ambient ``SYSTEM_ACTOR`` (unauthenticated/local) holds ``admin`` → sees
    all, so behaviour is unchanged until a real identity scopes it down.
    Degrades to the full set if agent-utilities predates the resolver, or if
    no verified actor context is bound yet (e.g. no identity middleware
    configured) — an entitlement-resolution failure must never break listing.
    """
    try:
        from agent_utilities.security.entitlements import (
            identity_scoped_resources,
        )

        return list(identity_scoped_resources(namespace, names))
    except Exception:
        return list(names)


async def _create_collection_action(
    client,
    db_type: str | None,
    collection_name: str | None,
    overwrite: bool | None,
    include_configured_directory: bool,
    document_paths: list[str] | None,
    document_contents: list[str] | None,
):
    document_directory, resolved_paths = resolve_document_inputs(
        configured_root=str(setting("DOCUMENT_DIRECTORY", "") or ""),
        include_configured_directory=include_configured_directory,
        relative_paths=document_paths,
        document_contents=document_contents,
    )
    kwargs = {
        "db_type": db_type,
        "collection_name": collection_name,
        "overwrite": overwrite,
        "document_directory": document_directory,
        "document_paths": resolved_paths,
        "document_contents": document_contents,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return await run_blocking(client.create_collection, **kwargs)


async def _add_documents_action(
    client,
    db_type: str | None,
    collection_name: str | None,
    include_configured_directory: bool,
    document_paths: list[str] | None,
    document_contents: list[str] | None,
):
    if (
        not include_configured_directory
        and not document_paths
        and not document_contents
    ):
        raise ValueError("At least one configured document input is required")
    document_directory, resolved_paths = resolve_document_inputs(
        configured_root=str(setting("DOCUMENT_DIRECTORY", "") or ""),
        include_configured_directory=include_configured_directory,
        relative_paths=document_paths,
        document_contents=document_contents,
    )
    kwargs = {
        "db_type": db_type,
        "collection_name": collection_name,
        "document_directory": document_directory,
        "document_paths": resolved_paths,
        "document_contents": document_contents,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return await run_blocking(client.add_documents, **kwargs)


async def _delete_collection_action(
    client, db_type: str | None, collection_name: str | None, confirm: bool | None
):
    kwargs = {
        "db_type": db_type,
        "collection_name": collection_name,
        "confirm": confirm,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return await run_blocking(client.delete_collection, **kwargs)


def _filter_entitled_collections(result):
    if not (isinstance(result, dict) and "collections" in result):
        return result
    entries = result["collections"]
    names = [
        str(entry.get("collection_name", "")) if isinstance(entry, dict) else str(entry)
        for entry in entries
    ]
    allowed = set(_entitled("collection", names))
    return {
        **result,
        "collections": [
            entry for entry, name in zip(entries, names, strict=True) if name in allowed
        ],
    }


async def _list_collections_action(client, db_type: str | None):
    result = await run_blocking(client.list_collections, db_type=db_type)
    return _filter_entitled_collections(result)


def register_collection_management_tools(mcp: FastMCP):
    @mcp.tool(
        tags={"collection_management"},
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        meta={
            "eg.annotations": {"modalities_in": ["text"], "modalities_out": ["text"]}
        },
    )
    async def vector_collection_management(
        action: Literal[
            "add_documents",
            "create_collection",
            "delete_collection",
            "list_collections",
        ] = Field(
            description="Action to perform. Must be one of: 'create_collection', 'add_documents', 'delete_collection', 'list_collections'"
        ),
        db_type: str | None = Field(default=None, description="db type"),
        collection_name: str | None = Field(
            default=None, description="collection name"
        ),
        overwrite: bool | None = Field(default=None, description="overwrite"),
        include_configured_directory: bool = Field(
            default=False,
            description="ingest the administrator-configured document root",
        ),
        document_paths: list[str] | None = Field(
            default=None,
            description="relative paths beneath the configured document root",
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
        resolved = resolve_action(
            action,
            (
                "create_collection",
                "add_documents",
                "delete_collection",
                "list_collections",
            ),
            service="vector-mcp",
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        selected_backend = _backend(db_type)
        selected_collection = _collection(collection_name)
        if action == "create_collection":
            return await _create_collection_action(
                client,
                selected_backend,
                selected_collection,
                overwrite,
                include_configured_directory,
                document_paths,
                document_contents,
            )
        if action == "add_documents":
            return await _add_documents_action(
                client,
                selected_backend,
                selected_collection,
                include_configured_directory,
                document_paths,
                document_contents,
            )
        if action == "delete_collection":
            return await _delete_collection_action(
                client, selected_backend, selected_collection, confirm
            )
        if action == "list_collections":
            return await _list_collections_action(client, selected_backend)
        raise ValueError("collection_action_invalid")


def _validate_search_question(question: str | None, number_results: int | None) -> None:
    if not question or len(question.encode("utf-8")) > 1_048_576:
        raise ValueError("Search question is invalid")
    if number_results is not None and not 1 <= number_results <= 1_000:
        raise ValueError("Result count is invalid")


def _validate_fusion_weights(
    semantic_weight: float | None, lexical_weight: float | None, rrf_k: int | None
) -> None:
    if semantic_weight is not None and not 0 <= semantic_weight <= 1:
        raise ValueError("Semantic weight is invalid")
    if lexical_weight is not None and not 0 <= lexical_weight <= 1:
        raise ValueError("Lexical weight is invalid")
    if rrf_k is not None and not 1 <= rrf_k <= 10_000:
        raise ValueError("RRF constant is invalid")


def _validate_search_request(
    question: str | None,
    number_results: int | None,
    semantic_weight: float | None,
    lexical_weight: float | None,
    rrf_k: int | None,
) -> None:
    _validate_search_question(question, number_results)
    _validate_fusion_weights(semantic_weight, lexical_weight, rrf_k)


async def _semantic_search_action(
    client, db_type, collection_name, question, number_results
):
    kwargs = {
        "db_type": db_type,
        "collection_name": collection_name,
        "question": question,
        "number_results": number_results,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return await run_blocking(client.semantic_search, **kwargs)


async def _lexical_search_action(
    client, db_type, collection_name, question, number_results
):
    kwargs = {
        "db_type": db_type,
        "collection_name": collection_name,
        "question": question,
        "number_results": number_results,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return await run_blocking(client.lexical_search, **kwargs)


@dataclass
class FusionWeights:
    """Bounds the RRF fusion tuning knobs into one boundary-crossing value."""

    semantic_weight: float | None
    lexical_weight: float | None
    rrf_k: int | None


async def _fused_search_action(
    client, db_type, collection_name, question, number_results, weights: FusionWeights
):
    kwargs = {
        "db_type": db_type,
        "collection_name": collection_name,
        "question": question,
        "number_results": number_results,
        "semantic_weight": weights.semantic_weight,
        "lexical_weight": weights.lexical_weight,
        "rrf_k": weights.rrf_k,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return await run_blocking(client.search, **kwargs)


def register_search_tools(mcp: FastMCP):
    @mcp.tool(tags={"search"})
    async def vector_search(
        action: str = Field(
            description="Action to perform. Must be one of: 'semantic_search', 'lexical_search', 'search'"
        ),
        db_type: str | None = Field(default=None, description="db type"),
        collection_name: str | None = Field(
            default=None, description="collection name"
        ),
        question: str | None = Field(default=None, description="question"),
        number_results: int | None = Field(default=None, description="number results"),
        semantic_weight: float | None = Field(
            default=None, description="semantic weight"
        ),
        lexical_weight: float | None = Field(
            default=None, description="lexical weight"
        ),
        rrf_k: int | None = Field(default=None, description="rrf k"),
        client=Depends(get_client),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict:
        """Manage search operations.

        Actions:
          - 'semantic_search': Retrieves and gathers related knowledge from the vector database instance using the question variable.
          - 'lexical_search': Performs indexed lexical or term-based retrieval.
          - 'search': Performs bounded RRF fusion over semantic and lexical retrieval.
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
        selected_backend = _backend(db_type)
        selected_collection = _collection(collection_name)
        _validate_search_request(
            question, number_results, semantic_weight, lexical_weight, rrf_k
        )
        if action == "semantic_search":
            return await _semantic_search_action(
                client, selected_backend, selected_collection, question, number_results
            )
        if action == "lexical_search":
            return await _lexical_search_action(
                client, selected_backend, selected_collection, question, number_results
            )
        if action == "search":
            weights = FusionWeights(semantic_weight, lexical_weight, rrf_k)
            return await _fused_search_action(
                client,
                selected_backend,
                selected_collection,
                question,
                number_results,
                weights,
            )
        raise ValueError("search_action_invalid")


def get_mcp_instance(command_args: list[str] | None = None) -> tuple[Any, ...]:
    """Initialize and return an embedded MCP instance.

    Embedded callers do not inherit unrelated host-process arguments. The CLI
    entry point passes its own arguments explicitly.
    """
    load_config()
    args, mcp, middlewares = create_mcp_server(
        name="vector-mcp MCP",
        version=__version__,
        instructions="vector-mcp MCP Server — Condensed Action-Routed Tools.",
        command_args=[] if command_args is None else command_args,
    )

    # /health is now registered by agent_connector_sdk.mcp.server.create_mcp_server itself.
    register_tool_surface(
        mcp,
        service="vector-mcp",
        tools_module=sys.modules[__name__],
    )

    for mw in middlewares:
        mcp.add_middleware(mw)
    return mcp, args, middlewares


def mcp_server() -> None:
    mcp, args, middlewares = get_mcp_instance(sys.argv[1:])
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
