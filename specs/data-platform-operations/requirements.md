# CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS-R001` | **Collection stats and get-by-id.** `vector_collection_management` returns count, dimension, embedding model and distance metric, and fetches one item by id. Touches `vector_mcp/mcp_server.py`. Moved from `MEDULLA-DATAPLAT-VOCAB-R009.2` (medulla). Caller chain: an MCP client calls `vector_collection_management` (vector_mcp/mcp_server.py:176) with a new `stats` action, which calls `VectorAPI` in `vector_mcp/vector_api.py` to read the collection. Shape: a `stats` action and a `get` by id action in `vector_collection_management` returning count, dimension, embedding model and distance metric; reuses the backend collection accessors. Proving test (planned): `tests/test_mcp_server.py::test_collection_stats_and_get_by_id`. Proving test (planned): `tests/test_mcp_server.py::test_collection_stats_and_get_by_id`. | stats action returns the four fields on a fixture collection. |
