# VECTOR — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

- [ ] VECTOR-R001.1: Verify vector-db schema matches embedding dimensions in vector_mcp/vectordb/ for medulla, postgresql, qdrant, mongodb.
- [ ] VECTOR-R001.2: Verify VectorAPI.semantic_search returns ranked results with correct score and metadata in vector_mcp/vector_api.py.
- [ ] VECTOR-R001.3: Verify vector_search MCP tool routes to VectorAPI.semantic_search in vector_mcp/mcp_server.py.
- [ ] VECTOR-R001.4: Verify dimension mismatch raises vector_dimension_mismatch; unsupported profile raises vector_database_type_unsupported in backend_policy.py.
- [ ] VECTOR-R001.5: Verify all four supported profiles pass protocol compliance and dimension validation in tests/test_protocol_compliance.py.
- [ ] VECTOR-R001.6: Verify vector_search tool integration end-to-end with real MCP invocation in tests/test_mcp_server.py.
- [x] VECTOR-R001.7: Install the store clients through the SDK `vector` and `vector-chroma` extras in pyproject.toml.
