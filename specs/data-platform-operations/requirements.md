# CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS-R001` | **Collection stats and get-by-id.** `vector_collection_management` returns count, dimension, embedding model and distance metric, and fetches one item by id. Touches `vector_mcp/mcp_server.py`. Moved from `MEDULLA-DATAPLAT-VOCAB-R009.2` (medulla). | stats action returns the four fields on a fixture collection. |
