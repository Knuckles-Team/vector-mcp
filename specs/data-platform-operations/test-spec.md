# CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS-R001` | `tests/test_mcp_server.py::test_collection_stats_and_get_by_id` (planned) | stats action returns the four fields on a fixture collection. |

## Negative and boundary cases

Each proving test above includes the refusal or error case that its row names. A vendor outage or an unauthorised call must return the SDK typed error.

## Quality and release proof

Run `uv run pytest tests` and `pre-commit run --all-files` from the repository root. A planned test is added by the code change that lands the row.
