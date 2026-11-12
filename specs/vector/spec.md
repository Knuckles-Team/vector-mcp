# VECTOR — Vector indexing and semantic search

Owner: vector-mcp. Cross-repo IDs: SDK transport contract, EG provider integration.
Public provenance: spec-audit closure SDK:8; addresses missing boundary definition for semantic-search operation.

## Purpose and user stories

Vector indexing and semantic-search operations require exact backend specification, dimension matching, and provider-specific handling of mismatch conditions. The MCP tool boundary must show real acceptance of index/query pairs across supported profiles and truthfully refuse unsupported combinations before transmission.

## Why a new spec

No prior spec owns semantic-search backend specification, dimension contracts, or provider-specific refusal. Meristem SDK specs define transport policy; this spec defines the served operation boundary.

## Scope and interfaces

Owned: vector_search MCP tool, VectorAPI backend routing, supported provider profiles (medulla, postgresql/pgvector, qdrant, mongodb), index/query dimension validation, unsupported-profile refusal.
Dependencies: Meristem SDK transport layer, configured vector-database factory, embedding model (shared create_embedding_model).
Non-goals: Collection lifecycle (create_collection), document metadata ingestion, full-text search.

## Open questions

- None.

## Delivery mapping

Component: vector-mcp connector

Code: ../../vector_mcp/__init__.py

## User Scenarios & Testing

### User Story 1 - Vector indexing and semantic search (Priority: P1)

The fleet operator and the engine rely on this behaviour of `vector-mcp`: VectorAPI indexes documents and returns semantic-search results with exact backend and dimension verification. It is proven independently of the other stories.

**Acceptance Scenarios**:

1. **Given** the `vector-mcp` connector built from the default branch, **When** `the planned test of this row` runs, **Then** integration tests pass dimension validation for each supported provider.

### User Story 2 - Model (Priority: P1)

The fleet operator and the engine rely on this behaviour of `vector-mcp`: Document index stores text in vector-db schema matching embedding dimensions for selected provider. It is proven independently of the other stories.

**Acceptance Scenarios**:

1. **Given** the `vector-mcp` connector built from the default branch, **When** `tests/test_protocol_compliance.py::test_index_schema_matches_embedding_dimension` runs, **Then** `test_protocol_compliance.py` verifies schema against medulla, qdrant, postgresql, and mongodb.

### User Story 3 - Behavior (Priority: P1)

The fleet operator and the engine rely on this behaviour of `vector-mcp`: Semantic search routes queries through provider backend, computing embedding and finding nearest neighbors. It is proven independently of the other stories.

**Acceptance Scenarios**:

1. **Given** the `vector-mcp` connector built from the default branch, **When** `tests/test_vector_search.py::test_semantic_search_succeeds` runs, **Then** `test_vector_search.py::test_semantic_search_succeeds` validates query result shape and ranking.

### User Story 4 - Wiring (Priority: P2)

The fleet operator and the engine rely on this behaviour of `vector-mcp`: VectorAPI.semantic_search depends on configured database factory; service layer routes through vector_search tool. It is proven independently of the other stories.

**Acceptance Scenarios**:

1. **Given** the `vector-mcp` connector built from the default branch, **When** `tests/test_mcp_server.py::test_vector_search_tools` runs, **Then** `test_mcp_server.py::test_vector_search_tool` traces call from MCP to provider.

### User Story 5 - Refusal (Priority: P2)

The fleet operator and the engine rely on this behaviour of `vector-mcp`: Query and index embedding dimension mismatch raises `vector_dimension_mismatch` typed refusal; unsupported profile raises `vector_database_type_unsupported`. It is proven independently of the other stories.

**Acceptance Scenarios**:

1. **Given** the `vector-mcp` connector built from the default branch, **When** `tests/test_error_handling.py::test_dimension_mismatch_and_unsupported_profile_are_typed_refusals` runs, **Then** `test_error_handling.py` verifies error type and message for each condition.

### User Story 6 - Evidence (Priority: P2)

The fleet operator and the engine rely on this behaviour of `vector-mcp`: Each supported provider profile includes tested query and index path using real tool invocation. It is proven independently of the other stories.

**Acceptance Scenarios**:

1. **Given** the `vector-mcp` connector built from the default branch, **When** `tests/test_protocol_compliance.py::test_full_real_call_path` runs, **Then** `test_protocol_compliance.py::test_full_real_call_path` covers medulla (default), postgresql, qdrant, mongodb.

### User Story 7 - Hot-path proof (Priority: P2)

The fleet operator and the engine rely on this behaviour of `vector-mcp`: MCP tool `vector_search(action=semantic_search, .)` serves vector_search entry point; integration test in `tests/test_mcp_server.py::test_vector_search_tool` validates path end-to-end. It is proven independently of the other stories.

**Acceptance Scenarios**:

1. **Given** the `vector-mcp` connector built from the default branch, **When** `tests/test_mcp_server.py::test_vector_search_tool` runs, **Then** full path test passes for each declared profile.

### User Story 8 - Store clients (Priority: P2)

The fleet operator and the engine rely on this behaviour of `vector-mcp`: The store extras `postgres`, `qdrant`, and `mongodb` install the SDK `vector` extra. It is proven independently of the other stories.

**Acceptance Scenarios**:

1. **Given** the `vector-mcp` connector built from the default branch, **When** `tests/test_store_extras.py::test_store_extras_come_from_the_sdk` runs, **Then** `tests/test_store_extras.py::test_store_extras_come_from_the_sdk` passes.

### Edge Cases

- The vendor service is unreachable or rejects the credential: the connector returns the SDK typed error and never a partial result.
- An argument outside the documented inputs is refused with the SDK typed refusal and causes no effect on the vendor service.

## Requirements

- **FR-001**: Vector indexing and semantic search. Rows: `VECTOR-R001`.
- **FR-002**: Model. Rows: `VECTOR-R001.1`.
- **FR-003**: Behavior. Rows: `VECTOR-R001.2`.
- **FR-004**: Wiring. Rows: `VECTOR-R001.3`.
- **FR-005**: Refusal. Rows: `VECTOR-R001.4`.
- **FR-006**: Evidence. Rows: `VECTOR-R001.5`.
- **FR-007**: Hot-path proof. Rows: `VECTOR-R001.6`.
- **FR-008**: Store clients. Rows: `VECTOR-R001.7`.

## Success Criteria

- **SC-001**: Integration tests pass dimension validation for each supported provider. Rows: `VECTOR-R001`.
- **SC-002**: `test_protocol_compliance.py` verifies schema against medulla, qdrant, postgresql, and mongodb. Rows: `VECTOR-R001.1`.
- **SC-003**: `test_vector_search.py::test_semantic_search_succeeds` validates query result shape and ranking. Rows: `VECTOR-R001.2`.
- **SC-004**: `test_mcp_server.py::test_vector_search_tool` traces call from MCP to provider. Rows: `VECTOR-R001.3`.
- **SC-005**: `test_error_handling.py` verifies error type and message for each condition. Rows: `VECTOR-R001.4`.
- **SC-006**: `test_protocol_compliance.py::test_full_real_call_path` covers medulla (default), postgresql, qdrant, mongodb. Rows: `VECTOR-R001.5`.
- **SC-007**: Full path test passes for each declared profile. Rows: `VECTOR-R001.6`.
- **SC-008**: `tests/test_store_extras.py::test_store_extras_come_from_the_sdk` passes. Rows: `VECTOR-R001.7`.

## Assumptions

- The `vector-mcp` package keeps the entry point and module layout named in the rows.
- The shared SDK supplies the transport, authentication, retry and certification helpers the rows reuse.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `VECTOR-R001` | plan.md Production caller chain | T-001 | planned |
| FR-002 | `VECTOR-R001.1` | plan.md Production caller chain | T-002 | tests/test_protocol_compliance.py::test_index_schema_matches_embedding_dimension |
| FR-003 | `VECTOR-R001.2` | plan.md Production caller chain | T-003 | tests/test_vector_search.py::test_semantic_search_succeeds |
| FR-004 | `VECTOR-R001.3` | plan.md Production caller chain | T-004 | tests/test_mcp_server.py::test_vector_search_tools |
| FR-005 | `VECTOR-R001.4` | plan.md Production caller chain | T-005 | tests/test_error_handling.py::test_dimension_mismatch_and_unsupported_profile_are_typed_refusals |
| FR-006 | `VECTOR-R001.5` | plan.md Production caller chain | T-006 | tests/test_protocol_compliance.py::test_full_real_call_path |
| FR-007 | `VECTOR-R001.6` | plan.md Production caller chain | T-007 | tests/test_mcp_server.py::test_vector_search_tool |
| FR-008 | `VECTOR-R001.7` | plan.md Production caller chain | T-008 | tests/test_store_extras.py::test_store_extras_come_from_the_sdk |
