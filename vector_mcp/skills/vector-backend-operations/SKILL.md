---
name: vector-backend-operations
skill_type: skill
description: >-
  Select and connect the right vector-store backend for the vector-mcp MCP server — chromadb, postgres/pgvector, qdrant, couchbase, or mongodb — and supply the correct db_type/connection parameters that every collection and search call needs. Use when the agent must decide which engine to target, wire up host/port/path/credentials, or move a workload between embedded and networked backends.
license: MIT
tags: [vector, rag, backend, chromadb, qdrant, postgres, mongodb, couchbase, mcp]
metadata:
  author: Knuckles-Team
  version: '0.1.0'
---
# Vector Backend Operations

Scope: Select and connect the right vector-store backend for the vector-mcp MCP server — chromadb, postgres/pgvector, qdrant, couchbase, or mongodb — and supply the correct db_type/connection parameters that every collection and search call needs. Use when the agent must decide which engine to target, wire up host/port/path/credentials, or move a workload between embedded and networked backends. Do NOT use for collection lifecycle (use vector-collection-management) or for running queries (use vector-hybrid-search).

Backend selection and connection wiring for the `vector-mcp` MCP server. Every collection and search tool takes the same `db_type` argument; the
connection details live in server configuration. This skill is the reference for
getting both right per engine.

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

## Steps

1. Pick the engine from the workload: run `vector_list_collections` with the candidate `db_type` and verify the call returns a collection list (possibly empty) instead of a connection error.
2. Confirm the matching extra is installed on the server (for example `vector-mcp[qdrant]`) by reading the error text of the probe call, and verify it is not an import error.
3. Match the connection parameters to the engine using the table below: `db_path` for embedded chromadb, `host`/`port` for networked engines, plus `db_name`/`username`/`password` for postgres, couchbase and mongodb. Verify no embedded and networked parameters are mixed.
4. When moving a corpus, call `vector_create_collection` and `vector_add_documents` on the target `db_type`. Verify with `vector_list_collections` that the new collection appears on the target backend.
5. Reuse the same `db_type` on every later `vector_*` call and verify results come from the intended store by listing its collections once more.

## Failure recovery

| Failure | Cause | Recovery |
|---------|-------|----------|
| Connection or import error on `vector_list_collections` | Backend extra not installed or backend unreachable | Install the matching `vector-mcp[...]` extra on the server and recheck `host`, `port` or `db_path` against the table above |
| Call hits an empty or wrong store | Embedded and networked parameters mixed, or credentials omitted | Supply only `db_path` for embedded chromadb, or `host`/`port` with `db_name`/`username`/`password` for networked engines |
| Request rejected for `db_type` | Value is not one of the supported strings | Use exactly `chromadb`, `postgres`, `qdrant`, `couchbase` or `mongodb` |

## Backends & connection parameters
`db_type` selects the engine; the other params depend on it:

| `db_type` | Typical params | Notes |
|-----------|----------------|-------|
| `chromadb` | `db_path` (embedded) or `host`/`port` | Zero-infra local default |
| `postgres` | `host`, `port`, `db_name`, `username`, `password` | pgvector-backed |
| `qdrant` | `host`, `port` | Fast ANN, `fastembed` extra |
| `couchbase` | `host`, `db_name`, `username`, `password` | |
| `mongodb` | `host`, `port`, `db_name`, `username`, `password` | Atlas Vector Search |

The `db_type` flows into every `vector_*` tool; connection details are server
configuration, so once the backend is chosen, reuse the same `db_type` on each call.

## Tools
Backend selection is the `db_type` argument shared by every tool:

| Tools | Purpose |
|-------|---------|
| `vector_list_collections`, `vector_create_collection`, `vector_add_documents`, `vector_delete_collection` | collection lifecycle on the chosen backend |
| `vector_semantic_search`, `vector_lexical_search`, `vector_search` | retrieval on the chosen backend |

## Recipes
Probe a backend by listing its collections: `vector_list_collections` with `db_type="chromadb"` (or `qdrant`, `postgres`, `mongodb`).

Migrate a corpus: `vector_list_collections` and read from the source backend, then `vector_create_collection` and `vector_add_documents` with the target `db_type` (see `vector-collection-management`).

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
