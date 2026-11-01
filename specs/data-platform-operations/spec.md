# CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS — Operations the data-platform vocabulary needs

Owner: vector-mcp. Cross-repo IDs: `MEDULLA-DATAPLAT-VOCAB-R009`.

## Purpose and user stories

The medulla data-platform vocabulary answers competency questions from connector facts. One question needs an operation this connector lacks. This spec owns that operation.

## Why a new spec

The existing specs in `specs/` cover packaged content and identity, not tool operations. None owns a new connector operation.

## Scope and interfaces

Owned: the one operation named in the row, touching `vector_mcp/mcp_server.py`. Non-goal: anything in the medulla vocabulary itself.

## Open questions

- None.

## Delivery mapping

Component: vector-mcp connector

Code: ../../vector_mcp/__init__.py

## User Scenarios & Testing

### User Story 1 - Collection stats and get-by-id (Priority: P1)

The fleet operator and the engine rely on this behaviour of `vector-mcp`: `vector_collection_management` returns count, dimension, embedding model and distance metric, and fetches one item by id. It is proven independently of the other stories.

**Acceptance Scenarios**:

1. **Given** the `vector-mcp` connector built from the default branch, **When** `tests/test_mcp_server.py::test_collection_stats_and_get_by_id` runs, **Then** stats action returns the four fields on a fixture collection.

### Edge Cases

- The vendor service is unreachable or rejects the credential: the connector returns the SDK typed error and never a partial result.
- An argument outside the documented inputs is refused with the SDK typed refusal and causes no effect on the vendor service.

## Requirements

- **FR-001**: Collection stats and get-by-id. Rows: `CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS-R001`.

## Success Criteria

- **SC-001**: stats action returns the four fields on a fixture collection. Rows: `CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS-R001`.

## Assumptions

- The `vector-mcp` package keeps the entry point and module layout named in the rows.
- The shared SDK supplies the transport, authentication, retry and certification helpers the rows reuse.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS-R001` | plan.md Production caller chain | T-001 | tests/test_mcp_server.py::test_collection_stats_and_get_by_id |
