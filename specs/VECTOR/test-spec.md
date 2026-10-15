# VECTOR — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | VECTOR-R001.1 | Contract | Epistemic-graph/postgresql/qdrant/mongodb backend selected | Index schema dimensions match embedding model output | test_protocol_compliance.py::test_schema_dimensions |
| T-002 | VECTOR-R001.2 | Integration | vector_search(action=semantic_search, query="...", db_type="qdrant") | Results ranked by similarity score with metadata present | test_vector_search.py::test_semantic_search_succeeds |
| T-003 | VECTOR-R001.3 | Contract | MCP tool vector_search invoked with semantic_search action | Call routed to VectorAPI.semantic_search; backend determined by db_type | test_mcp_server.py::test_vector_search_tool |
| T-004 | VECTOR-R001.4 | Contract | Query embedding (256-dim) vs. index (768-dim) mismatch | Raises vector_dimension_mismatch before backend call | test_error_handling.py::test_dimension_mismatch |
| T-005 | VECTOR-R001.4 | Contract | db_type="unsupported_backend" requested | Raises vector_database_type_unsupported typed refusal | test_error_handling.py::test_unsupported_profile |
| T-006 | VECTOR-R001.5 | Integration | Full vector_search path per profile (epistemic-graph, postgresql, qdrant, mongodb) | All profiles pass schema, dimension, and call-path validation | test_protocol_compliance.py::test_full_real_call_path |
| T-007 | VECTOR-R001.6 | Live | MCP client invokes vector_search tool end-to-end with real backend | Results returned or typed refusal raised; no external transmission on mismatch | test_mcp_server.py::test_vector_search_integration |

## Negative and boundary cases

- Empty query (acceptance: returns empty result set)
- Query with special characters (acceptance: embedding computed and search succeeds)
- Very large result set (acceptance: ranking completes within timeout)
- Backend unavailable (acceptance: backend-specific timeout refusal)
- Dimension mismatch (acceptance: typed refusal before call)
- Unsupported profile (acceptance: typed refusal at argument validation)
- Cancelled request mid-stream (acceptance: in-flight cancellation propagated)

## Quality and release proof

Run: `python -m pytest tests/test_protocol_compliance.py tests/test_vector_search.py tests/test_mcp_server.py tests/test_error_handling.py -xvs`
Run: `python -m compileall -q vector_mcp`
Run: `python scripts/security_contract.py --contract .security/security-contract.json validate`
