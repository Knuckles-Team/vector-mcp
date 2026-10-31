# VECTOR — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Existing system and reuse

VectorAPI abstracts provider selection and backend routing; vector_search MCP tool is the served boundary. VectorDBFactory and backend_policy already codify supported providers (medulla, postgresql, qdrant, mongodb). Embedding model shared via create_embedding_model. No new authority introduced.

## Architecture

MCP client → vector_search tool → VectorAPI.semantic_search → selected backend. Dimension validation occurs before provider invocation. Error cases (mismatch, unsupported type) raise typed refusals. Collection guards separate from query bodies.

## Interfaces and data model

vector_search tool accepts action=semantic_search, query text, and db_type parameter (defaults to medulla). Returns ranked results or refusal. Query embedding and index embedding dimensions must match; mismatch raises vector_dimension_mismatch.

## Live integration path

Client → mcp_server.py vector_search tool → VectorAPI.semantic_search → VectorDBFactory selected backend (medulla, postgresql, qdrant, mongodb) → provider semantic_search or engine ANN index → caller results.

## Quality and release gates

Protocol compliance: test_protocol_compliance.py validates schema for all declared profiles. MCP server integration: test_mcp_server.py verifies tool schema and call path. Error handling: test_error_handling.py covers dimension mismatch and unsupported type. Full path: real tool invocation per profile.

## Risks, alternatives, and decisions

Dimension mismatch is a runtime observable; validation at query time prevents silent failures. Collection creation is separate from query scope to preserve existing guards.

## VECTOR-R001

VECTOR-R001.1-6: vector_search(action=semantic_search) → VectorAPI.semantic_search → backend provider semantic_search → embedding dimension validation → query results or typed refusal.
