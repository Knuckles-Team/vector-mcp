---
name: vector-collection-management
skill_type: skill
description: >-
  Create, populate, list, and delete vector-store collections through the
  vector-mcp MCP server's collection tools (vector_create_collection, vector_add_documents, vector_delete_collection, vector_list_collections). Use when the agent
  must stand up a new RAG collection, ingest documents (from a directory, file
  paths/URLs, or raw text) into an existing collection, enumerate collections, or
  drop one — across any supported backend (chromadb, postgres/pgvector, qdrant,
  couchbase, mongodb). Do NOT use for running queries against a collection (use
  vector-hybrid-search) or for choosing/tuning a backend connection
  (vector-backend-operations).
license: MIT
tags: [vector, rag, collections, embeddings, mcp]
metadata:
  author: Knuckles-Team
  version: '0.1.0'
---
# Vector Collection Management

Lifecycle operations on vector-store **collections** — the named indexes that hold
embedded document chunks — via the `vector-mcp` MCP server. Four typed tools cover create / add / list / delete so the same call shape works
across every backend.

## When to use
- Create (or get-or-create) a collection to hold a corpus for RAG.
- Ingest documents into a collection — from a `document_directory`, a list of
  `document_paths` (files/URLs), or inline `document_contents` strings.
- List the collections that exist on a backend.
- Delete a collection you no longer need.

## When NOT to use
- Querying / retrieving from a collection → `vector-hybrid-search`.
- Picking a backend, wiring host/port/credentials, or comparing engines →
  `vector-backend-operations`.
- Persisting derived knowledge into the medulla KG → this is a vector
  backend, not a KG source; use the arbora tools for KG writes.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`vector-mcp`** MCP server. Every
tool takes the `db_type` (or falls back to the server default); connection details are server configuration:

| Parameter | Notes |
|-----------|-------|
| `db_type` | Backend: `chromadb`, `postgres`, `qdrant`, `couchbase`, `mongodb` |
| `db_path` | On-disk path (e.g. for embedded chromadb) |
| `host` / `port` | Networked backends |
| `db_name` / `username` / `password` | Auth for postgres / couchbase / mongodb |
| `collection_name` | Target collection (defaults to `memory` in the retriever) |

Embedding is handled server-side by the configured embedding model
(`scion-sdk` embedding client (OpenAI-compatible or Ollama)).

## Tools
| Tool | Purpose |
|------|---------|
| `vector_create_collection` | get-or-create a collection, optionally seeded from documents |
| `vector_add_documents` | add documents to an existing collection |
| `vector_list_collections` | list the collections the caller may see |
| `vector_delete_collection` | delete a collection (needs `confirm`) |

### Key parameters
- `overwrite` — on `create_collection`, replace an existing collection instead of
  get-or-create.
- `include_configured_directory` / `document_paths` / `document_contents` — the ingestion inputs for `vector_create_collection` and `vector_add_documents`; paths are relative to the configured document root.
- `confirm` — required guard on `delete_collection`.

## Recipes
Create a collection and seed it from the configured document root: `vector_create_collection` with `db_type="chromadb"`, `collection_name="handbook"`, `include_configured_directory=true`.

Add inline text to an existing collection: `vector_add_documents` with `collection_name="handbook"`, `document_contents=["VPN setup: connect to gw-hq before syncing."]`.

List collections on a backend: `vector_list_collections` with `db_type="qdrant"`.

Delete a collection (guarded): `vector_delete_collection` with `collection_name="handbook"`, `confirm=true`.

## Gotchas
- `vector_create_collection` is get-or-create by default; pass `overwrite:true` to rebuild
  from scratch — this drops existing vectors.
- `vector_delete_collection` needs `confirm:true`; without it the call is a no-op guard.
- Ingest inputs are supported one-at-a-time per call; loaders derive from
  `llama-index-readers-file` so PDFs/EPUBs/HTML are parsed, but very large
  directories should be chunked into batches to avoid long-running calls.
- The backend must be reachable with the given `db_type`/connection params or the
  call errors before any write.

## Related
- `vector-hybrid-search` — retrieve from the collections you populate here.
- `vector-backend-operations` — backend selection and connection details.
