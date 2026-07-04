---
name: vector-collection-management
description: >-
  Create, populate, list, and delete vector-store collections through the
  vector-mcp MCP server's vector_collection_management tool. Use when the agent
  must stand up a new RAG collection, ingest documents (from a directory, file
  paths/URLs, or raw text) into an existing collection, enumerate collections, or
  drop one — across any supported backend (chromadb, postgres/pgvector, qdrant,
  couchbase, mongodb). Do NOT use for running queries against a collection (use
  vector-hybrid-search) or for choosing/tuning a backend connection
  (vector-backend-operations).
license: MIT
tags: [vector, rag, collections, embeddings, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Vector Collection Management

Lifecycle operations on vector-store **collections** — the named indexes that hold
embedded document chunks — via the `vector-mcp` MCP server. One condensed,
action-routed tool covers create / add / list / delete so the same call shape works
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
- Persisting derived knowledge into the epistemic-graph KG → this is a vector
  backend, not a KG source; use the graph-os tools for KG writes.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`vector-mcp`** MCP server. Every
action takes the connection parameters inline (or falls back to server defaults):

| Parameter | Notes |
|-----------|-------|
| `db_type` | Backend: `chromadb`, `postgres`, `qdrant`, `couchbase`, `mongodb` |
| `db_path` | On-disk path (e.g. for embedded chromadb) |
| `host` / `port` | Networked backends |
| `db_name` / `username` / `password` | Auth for postgres / couchbase / mongodb |
| `collection_name` | Target collection (defaults to `memory` in the retriever) |

Embedding is handled server-side by the configured embedding model
(`agent-utilities[embeddings-openai]` or HuggingFace/sentence-transformers extras).

## Tools & actions
| Condensed tool | Actions |
|----------------|---------|
| `vector_collection_management` | `create_collection`, `add_documents`, `delete_collection`, `list_collections` |

### Key parameters
- `overwrite` — on `create_collection`, replace an existing collection instead of
  get-or-create.
- `document_directory` / `document_paths` / `document_contents` — the three
  mutually-usable ingestion inputs for `create_collection` and `add_documents`.
- `confirm` — required guard on `delete_collection`.

## Recipes
Create a collection and seed it from a directory:
```json
{"action":"create_collection","db_type":"chromadb","db_path":"./chroma","collection_name":"handbook","document_directory":"./docs"}
```
Add inline text to an existing collection:
```json
{"action":"add_documents","db_type":"chromadb","db_path":"./chroma","collection_name":"handbook","document_contents":["VPN setup: connect to gw-hq before syncing."]}
```
List collections on a backend:
```json
{"action":"list_collections","db_type":"qdrant","host":"qdrant","port":"6333"}
```
Delete a collection (guarded):
```json
{"action":"delete_collection","db_type":"chromadb","db_path":"./chroma","collection_name":"handbook","confirm":true}
```

## Gotchas
- `create_collection` is get-or-create by default; pass `overwrite:true` to rebuild
  from scratch — this drops existing vectors.
- `delete_collection` needs `confirm:true`; without it the call is a no-op guard.
- Ingest inputs are supported one-at-a-time per call; loaders derive from
  `llama-index-readers-file` so PDFs/EPUBs/HTML are parsed, but very large
  directories should be chunked into batches to avoid long-running calls.
- The backend must be reachable with the given `db_type`/connection params or the
  call errors before any write.

## Related
- `vector-hybrid-search` — retrieve from the collections you populate here.
- `vector-backend-operations` — backend selection and connection details.
