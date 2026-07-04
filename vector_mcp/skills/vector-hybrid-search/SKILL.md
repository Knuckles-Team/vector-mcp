---
name: vector-hybrid-search
description: >-
  Retrieve knowledge from a vector-store collection via the vector-mcp MCP
  server's vector_search tool — semantic (ANN) search, lexical BM25 search, or a
  hybrid of the two fused with Reciprocal Rank Fusion. Use when the agent must
  answer a question from an indexed corpus, pull top-k relevant chunks for RAG
  context, or tune the semantic-vs-lexical balance of retrieval. Do NOT use for
  creating collections or ingesting documents (use vector-collection-management)
  or for backend connection setup (vector-backend-operations).
license: MIT
tags: [vector, rag, search, semantic, bm25, hybrid, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Vector Hybrid Search

Retrieval over vector-store **collections** through the `vector-mcp` MCP server.
One condensed tool exposes three retrieval strategies: dense semantic search,
sparse lexical BM25 search, and a hybrid that fuses both rankings.

## When to use
- Answer a natural-language question from an indexed corpus (grab RAG context).
- Fetch the top-k most relevant chunks for a query.
- Do keyword/term-exact retrieval where BM25 beats embeddings (codes, IDs, rare
  tokens).
- Balance recall (semantic) against precision on exact terms (lexical) via hybrid
  fusion.

## When NOT to use
- Creating / populating / deleting collections → `vector-collection-management`.
- Selecting or configuring the backend engine → `vector-backend-operations`.
- General web search or KG queries — this only searches the named collection.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`vector-mcp`** MCP server. The
collection must already exist and be populated (see
`vector-collection-management`). Connection params (`db_type`, `db_path`,
`host`/`port`, `db_name`, `username`, `password`, `collection_name`) are passed
inline, matching the collection you want to query.

## Tools & actions
| Condensed tool | Actions |
|----------------|---------|
| `vector_search` | `semantic_search`, `lexical_search`, `search` (hybrid) |

### Key parameters
- `question` — the query text (required for every action).
- `number_results` — top-k to return.
- `semantic_weight` / `bm25_weight` — leg weights for the hybrid `search` action.
- `rrf_k` — the Reciprocal Rank Fusion constant that merges the two rankings.

## Recipes
Semantic (vector) search:
```json
{"action":"semantic_search","db_type":"chromadb","db_path":"./chroma","collection_name":"handbook","question":"how do I connect to the VPN?","number_results":5}
```
Lexical BM25 search (term-exact):
```json
{"action":"lexical_search","db_type":"chromadb","db_path":"./chroma","collection_name":"handbook","question":"gw-hq","number_results":5}
```
Hybrid search, semantic-leaning fusion:
```json
{"action":"search","db_type":"chromadb","db_path":"./chroma","collection_name":"handbook","question":"vpn gateway hostname","number_results":8,"semantic_weight":0.7,"bm25_weight":0.3,"rrf_k":60}
```

## Gotchas
- `question` is required; an empty/missing question returns nothing useful.
- Results carry a score whose meaning depends on the action — semantic returns a
  distance (lower = closer), BM25 a relevance score (higher = better), hybrid a
  fused RRF score; don't compare scores across actions.
- Hybrid weights are only honored by the `search` action; they're ignored by the
  pure `semantic_search` / `lexical_search` actions.
- A typical `rrf_k` is 60; smaller `rrf_k` sharpens toward top ranks, larger
  flattens the fusion.
- Retrieval only covers the specified `collection_name` on the specified backend —
  point the connection params at the collection you actually populated.

## Related
- `vector-collection-management` — build and load the collections you search here.
- `vector-backend-operations` — backend/engine selection and connection tuning.
