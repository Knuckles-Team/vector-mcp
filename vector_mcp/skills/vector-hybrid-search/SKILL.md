---
name: vector-hybrid-search
skill_type: skill
description: >-
  Retrieve knowledge from a vector-store collection via the vector-mcp MCP
  server's vector_semantic_search, vector_lexical_search and vector_search tools — semantic (ANN) search, lexical BM25 search, or a
  hybrid of the two fused with Reciprocal Rank Fusion. Use when the agent must
  answer a question from an indexed corpus, pull top-k relevant chunks for RAG
  context, or tune the semantic-vs-lexical balance of retrieval. Do NOT use for
  creating collections or ingesting documents (use vector-collection-management)
  or for backend connection setup (vector-backend-operations).
license: MIT
tags: [vector, rag, search, semantic, bm25, hybrid, mcp]
metadata:
  author: Knuckles-Team
  version: '0.1.0'
---
# Vector Hybrid Search

Retrieval over vector-store **collections** through the `vector-mcp` MCP server.
Three tools expose three retrieval strategies: dense semantic search,
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
`vector-collection-management`). The tools take `db_type` and `collection_name`; connection details are server configuration.

## Tools
| Tool | Strategy |
|------|----------|
| `vector_semantic_search` | dense ANN retrieval |
| `vector_lexical_search` | BM25 term retrieval |
| `vector_search` | hybrid reciprocal-rank fusion of both |

### Key parameters
- `question` — the query text (required for every tool).
- `number_results` — top-k to return.
- `semantic_weight` / `lexical_weight` — leg weights for the hybrid `vector_search` tool.
- `rrf_k` — the Reciprocal Rank Fusion constant that merges the two rankings.

## Recipes
Semantic: `vector_semantic_search` with `collection_name="handbook"`, `question="how do I connect to the VPN?"`, `number_results=5`.

Lexical for exact tokens: `vector_lexical_search` with `collection_name="handbook"`, `question="gw-hq"`, `number_results=5`.

Hybrid: `vector_search` with `collection_name="handbook"`, `question="vpn gateway hostname"`, `number_results=8`, `semantic_weight=0.7`, `lexical_weight=0.3`, `rrf_k=60`.

## Gotchas
- `question` is required; an empty/missing question returns nothing useful.
- Results carry a score whose meaning depends on the tool — semantic returns a
  distance (lower = closer), BM25 a relevance score (higher = better), hybrid a
  fused RRF score; don't compare scores across tools.
- Hybrid weights are only honored by `vector_search`; the pure `vector_semantic_search` / `vector_lexical_search` tools ignore them.
- A typical `rrf_k` is 60; smaller `rrf_k` sharpens toward top ranks, larger
  flattens the fusion.
- Retrieval only covers the specified `collection_name` on the specified backend —
  point the connection params at the collection you actually populated.

## Related
- `vector-collection-management` — build and load the collections you search here.
- `vector-backend-operations` — backend/engine selection and connection tuning.
