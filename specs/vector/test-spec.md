# VECTOR — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `VECTOR-R001` | covered by the child rows | Integration tests pass dimension validation for each supported provider. |
| T-002 | `VECTOR-R001.1` | `tests/test_protocol_compliance.py::test_index_schema_matches_embedding_dimension` (planned) | `test_protocol_compliance.py` verifies schema against medulla, qdrant, postgresql, and mongodb. |
| T-003 | `VECTOR-R001.2` | `tests/test_vector_search.py::test_semantic_search_succeeds` (planned) | `test_vector_search.py::test_semantic_search_succeeds` validates query result shape and ranking. |
| T-004 | `VECTOR-R001.3` | `tests/test_mcp_server.py::test_vector_search_tools` | `test_mcp_server.py::test_vector_search_tool` traces call from MCP to provider. |
| T-005 | `VECTOR-R001.4` | `tests/test_error_handling.py::test_dimension_mismatch_and_unsupported_profile_are_typed_refusals` (planned) | `test_error_handling.py` verifies error type and message for each condition. |
| T-006 | `VECTOR-R001.5` | `tests/test_protocol_compliance.py::test_full_real_call_path` (planned) | `test_protocol_compliance.py::test_full_real_call_path` covers medulla (default), postgresql, qdrant, mongodb. |
| T-007 | `VECTOR-R001.6` | `tests/test_mcp_server.py::test_vector_search_tool` (planned) | Full path test passes for each declared profile. |
| T-008 | `VECTOR-R001.7` | `tests/test_store_extras.py::test_store_extras_come_from_the_sdk` | `tests/test_store_extras.py::test_store_extras_come_from_the_sdk` passes. |

## Negative and boundary cases

Each proving test above includes the refusal or error case that its row names. A vendor outage or an unauthorised call must return the SDK typed error.

## Quality and release proof

Run `uv run pytest tests` and `pre-commit run --all-files` from the repository root. A planned test is added by the code change that lands the row.
