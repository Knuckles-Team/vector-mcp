"""Surface contracts: tool use, links, data locations and business roles."""

from scion_sdk.surface_contracts import (
    BusinessRole,
    DataLocation,
    Refusal,
    SurfaceLink,
    ToolStep,
    ToolUseContract,
)

CONTRACTS = (
    ToolUseContract(
        task="inventory-collections",
        description="List the collections the caller may see on a backend.",
        steps=(
            ToolStep(
                tool="vector_list_collections",
                params=("db_type",),
                result=(
                    "collections",
                    "collection_name",
                ),
                refusals=(
                    Refusal(
                        code="UNAVAILABLE", when="the selected backend is not available"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="build-collection",
        description="Create a collection, load documents into it and list it back.",
        steps=(
            ToolStep(
                tool="vector_create_collection",
                params=(
                    "db_type",
                    "collection_name",
                    "overwrite",
                    "include_configured_directory",
                    "document_paths",
                    "document_contents",
                ),
                result=(
                    "collection_name",
                    "status",
                ),
                refusals=(
                    Refusal(
                        code="INVALID_ARGUMENT",
                        when="the collection name or document input is invalid",
                    ),
                ),
            ),
            ToolStep(
                tool="vector_add_documents",
                params=(
                    "db_type",
                    "collection_name",
                    "include_configured_directory",
                    "document_paths",
                    "document_contents",
                ),
                result=(
                    "collection_name",
                    "added",
                ),
                refusals=(
                    Refusal(code="NOT_FOUND", when="the collection does not exist"),
                ),
            ),
            ToolStep(
                tool="vector_list_collections",
                params=("db_type",),
                result=("collections",),
                refusals=(
                    Refusal(
                        code="ACCESS_DENIED",
                        when="the credential lacks access to the operation",
                    ),
                    Refusal(
                        code="INVALID_ARGUMENT",
                        when="the upstream system rejects the input",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="retire-collection",
        description="Delete a collection after listing it and getting approval.",
        steps=(
            ToolStep(
                tool="vector_list_collections",
                params=("db_type",),
                result=("collections",),
            ),
            ToolStep(
                tool="vector_delete_collection",
                params=(
                    "db_type",
                    "collection_name",
                    "confirm",
                ),
                result=(
                    "collection_name",
                    "status",
                ),
                refusals=(
                    Refusal(code="CONFIRMATION_REQUIRED", when="confirm was not set"),
                    Refusal(code="NOT_FOUND", when="the collection does not exist"),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="answer-from-corpus",
        description="Retrieve ranked chunks for a question, fused when both signals matter.",
        steps=(
            ToolStep(
                tool="vector_search",
                params=(
                    "db_type",
                    "collection_name",
                    "question",
                    "number_results",
                    "semantic_weight",
                    "lexical_weight",
                    "rrf_k",
                ),
                result=(
                    "results",
                    "score",
                ),
                refusals=(
                    Refusal(code="NOT_FOUND", when="the collection does not exist"),
                ),
            ),
            ToolStep(
                tool="vector_semantic_search",
                params=(
                    "db_type",
                    "collection_name",
                    "question",
                    "number_results",
                ),
                result=(
                    "results",
                    "score",
                ),
                refusals=(
                    Refusal(code="NOT_FOUND", when="the collection does not exist"),
                ),
            ),
            ToolStep(
                tool="vector_lexical_search",
                params=(
                    "db_type",
                    "collection_name",
                    "question",
                    "number_results",
                ),
                result=(
                    "results",
                    "score",
                ),
                refusals=(
                    Refusal(code="NOT_FOUND", when="the collection does not exist"),
                ),
            ),
        ),
    ),
)

LINKS = (
    SurfaceLink(
        task="inventory-collections",
        tools=("vector_list_collections",),
        prompts=(
            "vector-mcp-operator",
            "vector-mcp-knowledge-librarian",
            "vector-mcp-research-analyst",
        ),
        skills=(
            "vector-backend-operations",
            "vector-collection-management",
            "vector-mcp-operations",
        ),
    ),
    SurfaceLink(
        task="build-collection",
        tools=(
            "vector_create_collection",
            "vector_add_documents",
            "vector_list_collections",
        ),
        prompts=(
            "vector-mcp-operator",
            "vector-mcp-knowledge-librarian",
        ),
        skills=("vector-collection-management",),
    ),
    SurfaceLink(
        task="retire-collection",
        tools=(
            "vector_list_collections",
            "vector_delete_collection",
        ),
        prompts=(
            "vector-mcp-operator",
            "vector-mcp-knowledge-librarian",
        ),
        skills=("vector-collection-management",),
    ),
    SurfaceLink(
        task="answer-from-corpus",
        tools=(
            "vector_search",
            "vector_semantic_search",
            "vector_lexical_search",
        ),
        prompts=(
            "vector-mcp-operator",
            "vector-mcp-research-analyst",
        ),
        skills=("vector-hybrid-search",),
    ),
)

LOCATIONS = (
    DataLocation(
        "vector-collections",
        (
            "VectorCollection",
            "VectorChunk",
        ),
        "DATABASE_TYPE",
        "vector_list_collections",
        federation_key="collection_name",
        route_param="db_type",
    ),
)

ROLES = (
    BusinessRole(
        "knowledge librarian",
        tasks=(
            "inventory-collections",
            "build-collection",
            "retire-collection",
        ),
    ),
    BusinessRole(
        "research analyst",
        tasks=(
            "inventory-collections",
            "answer-from-corpus",
        ),
    ),
)
