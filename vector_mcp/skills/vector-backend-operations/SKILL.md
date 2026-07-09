---
name: vector-backend-operations
skill_type: skill
description: >-
  Select and connect the right vector-store backend for the vector-mcp MCP server
  — chromadb, postgres/pgvector, qdrant, couchbase, or mongodb — and supply the
  correct db_type/connection parameters that every collection and search call
  needs. Use when the agent must decide which engine to target, wire up
  host/port/path/credentials, or move a workload between embedded and networked
  backends. Do NOT use for collection lifecycle (use
  vector-collection-management) or for running queries (use vector-hybrid-search).
license: MIT
tags: [vector, rag, backend, chromadb, qdrant, postgres, mongodb, couchbase, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Vector Backend Operations

Backend selection and connection wiring for the `vector-mcp` MCP server. Both the
`vector_collection_management` and `vector_search` tools take the same `db_type` +
connection parameters — this skill is the reference for getting those right per
engine.

## When to use
- Choose a vector backend for a workload (embedded vs. networked, dev vs. prod).
- Assemble the correct connection parameters for a chosen `db_type`.
- Migrate a corpus between backends (e.g. local chromadb → hosted qdrant).
- Diagnose "backend unreachable / wrong params" failures before running collection
  or search calls.

## When NOT to use
- Creating/loading/deleting collections → `vector-collection-management`.
- Semantic / lexical / hybrid retrieval → `vector-hybrid-search`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`vector-mcp`** MCP server. The
relevant backend extra must be installed on the server (`vector-mcp[chromadb]`,
`[postgres]`, `[qdrant]`, `[couchbase]`, `[mongodb]`, or `[all]`).

## Backends & connection parameters
`db_type` selects the engine; the other params depend on it:

| `db_type` | Typical params | Notes |
|-----------|----------------|-------|
| `chromadb` | `db_path` (embedded) or `host`/`port` | Zero-infra local default |
| `postgres` | `host`, `port`, `db_name`, `username`, `password` | pgvector-backed |
| `qdrant` | `host`, `port` | Fast ANN, `fastembed` extra |
| `couchbase` | `host`, `db_name`, `username`, `password` | |
| `mongodb` | `host`, `port`, `db_name`, `username`, `password` | Atlas Vector Search |

The same params flow into `vector_collection_management` and `vector_search`, so
once you know the backend shape, reuse it across every call.

## Tools & actions
Backend selection is not a standalone action — it is the `db_type` + connection
arguments shared by both tools:

| Condensed tool | Actions using these params |
|----------------|----------------------------|
| `vector_collection_management` | `create_collection`, `add_documents`, `list_collections`, `delete_collection` |
| `vector_search` | `semantic_search`, `lexical_search`, `search` |

## Recipes
Probe an embedded chromadb backend by listing its collections:
```json
{"action":"list_collections","db_type":"chromadb","db_path":"./chroma"}
```
Point at a networked qdrant:
```json
{"action":"list_collections","db_type":"qdrant","host":"qdrant","port":"6333"}
```
Postgres/pgvector connection shape:
```json
{"action":"list_collections","db_type":"postgres","host":"pg","port":"5432","db_name":"rag","username":"rag","password":"<secret>"}
```
Migrate a corpus: `list_collections` + read from the source backend, then
`create_collection` + `add_documents` against the target `db_type` (see
`vector-collection-management`).

## Gotchas
- Embedded chromadb wants `db_path`; networked engines want `host`/`port` — mixing
  them silently targets the wrong store or fails to connect.
- The backend extra must be installed server-side; a missing extra surfaces as an
  import/connection error, not a bad-param error.
- Credentials for postgres/couchbase/mongodb are required — omitting them falls
  back to server defaults which may point elsewhere.
- Collections do not migrate automatically between backends; re-embed and re-ingest
  when moving engines (embedding models/dimensions must match on the target).
- The `db_type` string must be one of the supported set exactly; unknown values
  are rejected.

## Related
- `vector-collection-management` — collection lifecycle on the chosen backend.
- `vector-hybrid-search` — retrieval once the backend and collection are set.
