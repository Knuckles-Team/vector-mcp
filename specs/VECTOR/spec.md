# VECTOR — Vector indexing and semantic search

Owner: vector-mcp. Cross-repo IDs: SDK transport contract, EG provider integration.
Public provenance: spec-audit closure SDK:8; addresses missing boundary definition for semantic-search operation.

## Purpose and user stories

Vector indexing and semantic-search operations require exact backend specification, dimension matching, and provider-specific handling of mismatch conditions. The MCP tool boundary must show real acceptance of index/query pairs across supported profiles and truthfully refuse unsupported combinations before transmission.

## Requirements and acceptance

- VECTOR-R001: Vector indexing and semantic search backend behavior, dimension matching, and mismatch refusal.

## Why a new spec

No prior spec owns semantic-search backend specification, dimension contracts, or provider-specific refusal. Agent-utilities SDK specs define transport policy; this spec defines the served operation boundary.

## Scope and interfaces

Owned: vector_search MCP tool, VectorAPI backend routing, supported provider profiles (medulla, postgresql/pgvector, qdrant, mongodb), index/query dimension validation, unsupported-profile refusal.
Dependencies: Agent-utilities SDK transport layer, configured vector-database factory, embedding model (shared create_embedding_model).
Non-goals: Collection lifecycle (create_collection), document metadata ingestion, full-text search.

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| VECTOR-R001 | plan.md § Vector-R001 | test_protocol_compliance.py |  |

## Open questions

- None.
