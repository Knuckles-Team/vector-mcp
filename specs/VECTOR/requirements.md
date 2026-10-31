# VECTOR — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A child row `.n` is one five-minute unit of work. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `VECTOR-R001` | **Vector indexing and semantic search.** VectorAPI indexes documents and returns semantic-search results with exact backend and dimension verification. | Integration tests pass dimension validation for each supported provider. |
| `VECTOR-R001.1` | **Model.** Document index stores text in vector-db schema matching embedding dimensions for selected provider. | `test_protocol_compliance.py` verifies schema against medulla, qdrant, postgresql, and mongodb. |
| `VECTOR-R001.2` | **Behavior.** Semantic search routes queries through provider backend, computing embedding and finding nearest neighbors. | `test_vector_search.py::test_semantic_search_succeeds` validates query result shape and ranking. |
| `VECTOR-R001.3` | **Wiring.** VectorAPI.semantic_search depends on configured database factory; service layer routes through vector_search tool. | `test_mcp_server.py::test_vector_search_tool` traces call from MCP to provider. |
| `VECTOR-R001.4` | **Refusal.** Query and index embedding dimension mismatch raises `vector_dimension_mismatch` typed refusal; unsupported profile raises `vector_database_type_unsupported`. | `test_error_handling.py` verifies error type and message for each condition. |
| `VECTOR-R001.5` | **Evidence.** Each supported provider profile includes tested query and index path using real tool invocation. | `test_protocol_compliance.py::test_full_real_call_path` covers medulla (default), postgresql, qdrant, mongodb. |
| `VECTOR-R001.6` | **Hot-path proof.** MCP tool `vector_search(action=semantic_search, ...)` serves vector_search entry point; integration test in `tests/test_mcp_server.py::test_vector_search_tool` validates path end-to-end. | Full path test passes for each declared profile. |
| `VECTOR-R001.7` | **Store clients.** The store extras `postgres`, `qdrant`, and `mongodb` install the SDK `vector` extra. The `chromadb` extra installs the SDK `vector-chroma` extra. The connector pins no vector-store client itself. | `tests/test_store_extras.py::test_store_extras_come_from_the_sdk` passes. |
