#!/usr/bin/python
import warnings

from fastmcp import FastMCP
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
from typing import Any

from scion_sdk.config import load_config, setting
from scion_sdk.mcp.concurrency import run_blocking
from scion_sdk.mcp.server import create_mcp_server
from starlette.requests import Request
from starlette.responses import JSONResponse

from vector_mcp import __version__
from vector_mcp.backend_policy import ensure_backend_available
from vector_mcp.document_inputs import resolve_document_inputs
from vector_mcp.vector_api import get_client

logger = get_logger(name="vector-mcp")
logger.setLevel(logging.INFO)

_COLLECTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")


def _backend(value: str | None) -> str:
    configured = value or str(setting("DATABASE_TYPE", "medulla") or "medulla")
    return ensure_backend_available(configured)


def _collection(value: str | None) -> str | None:
    if value is None:
        return None
    if not _COLLECTION_NAME.fullmatch(value):
        raise ValueError("Collection name is invalid")
    return value


def _entitled(namespace: str, names: list[str]) -> list[str]:
    """Filter ``names`` to what the calling identity may reach.

    Routes the set through scion-sdk' shared identity-scoped resolver
    (CONCEPT:AU-OS.identity.identity-scoped-resource-autoload): a caller's
    Okta/Keycloak groups decide which resources they see by default. The
    ambient ``SYSTEM_ACTOR`` (unauthenticated/local) holds ``admin`` → sees
    all, so behaviour is unchanged until a real identity scopes it down.
    Degrades to the full set if scion-sdk predates the resolver, or if
    no verified actor context is bound yet (e.g. no identity middleware
    configured) — an entitlement-resolution failure must never break listing.
    """
    try:
        from scion_sdk.entitlements import identity_scoped_resources

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


_DB_TYPE = Field(default=None, description="db type")
_COLLECTION_NAME_FIELD = Field(default=None, description="collection name")
_DOC_ROOT = Field(
    default=False, description="ingest the administrator-configured document root"
)
_DOC_PATHS = Field(
    default=None, description="relative paths beneath the configured document root"
)
_DOC_CONTENTS = Field(default=None, description="document contents")


def register_collection_management_tools(mcp: FastMCP):
    """Register the collection tools, one typed tool per operation."""

    @mcp.tool(tags={"collection_management", "operation", "granular"})
    async def vector_create_collection(
        db_type: str | None = _DB_TYPE,
        collection_name: str | None = _COLLECTION_NAME_FIELD,
        overwrite: bool | None = Field(default=None, description="overwrite"),
        include_configured_directory: bool = _DOC_ROOT,
        document_paths: list[str] | None = _DOC_PATHS,
        document_contents: list[str] | None = _DOC_CONTENTS,
        client=Depends(get_client),
    ) -> dict:
        """Creates a new collection or retrieves an existing one in the vector database."""
        return await _create_collection_action(
            client,
            _backend(db_type),
            _collection(collection_name),
            overwrite,
            include_configured_directory,
            document_paths,
            document_contents,
        )

    @mcp.tool(tags={"collection_management", "operation", "granular"})
    async def vector_add_documents(
        db_type: str | None = _DB_TYPE,
        collection_name: str | None = _COLLECTION_NAME_FIELD,
        include_configured_directory: bool = _DOC_ROOT,
        document_paths: list[str] | None = _DOC_PATHS,
        document_contents: list[str] | None = _DOC_CONTENTS,
        client=Depends(get_client),
    ) -> dict:
        """Adds documents to an existing collection in the vector database."""
        return await _add_documents_action(
            client,
            _backend(db_type),
            _collection(collection_name),
            include_configured_directory,
            document_paths,
            document_contents,
        )

    @mcp.tool(
        tags={"collection_management", "operation", "granular"},
        annotations={
            "title": "Delete Collection",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def vector_delete_collection(
        db_type: str | None = _DB_TYPE,
        collection_name: str | None = _COLLECTION_NAME_FIELD,
        confirm: bool | None = Field(
            default=None, description="must be true to delete the collection"
        ),
        client=Depends(get_client),
    ) -> dict:
        """Deletes a collection from the vector database."""
        return await _delete_collection_action(
            client, _backend(db_type), _collection(collection_name), confirm
        )

    @mcp.tool(
        tags={"collection_management", "operation", "granular"},
        annotations={
            "title": "List Collections",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def vector_list_collections(
        db_type: str | None = _DB_TYPE,
        client=Depends(get_client),
    ) -> dict:
        """Lists the collections of the caller's identity in the vector database."""
        return await _list_collections_action(client, _backend(db_type))


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


_QUESTION = Field(default=None, description="question")
_NUMBER_RESULTS = Field(default=None, description="number results")
_READ_ONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


def register_search_tools(mcp: FastMCP):
    """Register the search tools, one typed tool per retrieval mode."""

    @mcp.tool(
        tags={"search", "operation", "granular"},
        annotations={"title": "Semantic Search", **_READ_ONLY_ANNOTATIONS},
    )
    async def vector_semantic_search(
        db_type: str | None = _DB_TYPE,
        collection_name: str | None = _COLLECTION_NAME_FIELD,
        question: str | None = _QUESTION,
        number_results: int | None = _NUMBER_RESULTS,
        client=Depends(get_client),
    ) -> dict:
        """Retrieves and gathers related knowledge from the vector database instance using the question variable."""
        _validate_search_request(question, number_results, None, None, None)
        return await _semantic_search_action(
            client,
            _backend(db_type),
            _collection(collection_name),
            question,
            number_results,
        )

    @mcp.tool(
        tags={"search", "operation", "granular"},
        annotations={"title": "Lexical Search", **_READ_ONLY_ANNOTATIONS},
    )
    async def vector_lexical_search(
        db_type: str | None = _DB_TYPE,
        collection_name: str | None = _COLLECTION_NAME_FIELD,
        question: str | None = _QUESTION,
        number_results: int | None = _NUMBER_RESULTS,
        client=Depends(get_client),
    ) -> dict:
        """Performs indexed lexical or term-based retrieval."""
        _validate_search_request(question, number_results, None, None, None)
        return await _lexical_search_action(
            client,
            _backend(db_type),
            _collection(collection_name),
            question,
            number_results,
        )

    @mcp.tool(
        tags={"search", "operation", "granular"},
        annotations={"title": "Fused Search", **_READ_ONLY_ANNOTATIONS},
    )
    async def vector_search(
        db_type: str | None = _DB_TYPE,
        collection_name: str | None = _COLLECTION_NAME_FIELD,
        question: str | None = _QUESTION,
        number_results: int | None = _NUMBER_RESULTS,
        semantic_weight: float | None = Field(
            default=None, description="semantic weight"
        ),
        lexical_weight: float | None = Field(
            default=None, description="lexical weight"
        ),
        rrf_k: int | None = Field(default=None, description="rrf k"),
        client=Depends(get_client),
    ) -> dict:
        """Performs bounded RRF fusion over semantic and lexical retrieval."""
        _validate_search_request(
            question, number_results, semantic_weight, lexical_weight, rrf_k
        )
        return await _fused_search_action(
            client,
            _backend(db_type),
            _collection(collection_name),
            question,
            number_results,
            FusionWeights(semantic_weight, lexical_weight, rrf_k),
        )


def get_mcp_instance(command_args: list[str] | None = None) -> tuple[Any, ...]:
    """Initialize and return an embedded MCP instance.

    Embedded callers do not inherit unrelated host-process arguments. The CLI
    entry point passes its own arguments explicitly.
    """
    load_config()
    args, mcp, middlewares = create_mcp_server(
        name="vector-mcp MCP",
        version=__version__,
        instructions="vector-mcp MCP Server — one typed tool per operation.",
        command_args=[] if command_args is None else command_args,
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})

    for toggle, registrar in (
        ("COLLECTION_MANAGEMENTTOOL", register_collection_management_tools),
        ("SEARCHTOOL", register_search_tools),
    ):
        if setting(toggle, True):
            registrar(mcp)

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
