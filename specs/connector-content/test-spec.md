# CONN-VECTOR-MCP-CONNECTOR-CONTENT — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Proving test | Expected observation |
|---|---|---|---|
| T-001 | `CONN-VECTOR-MCP-CONNECTOR-CONTENT-R001` | `tests/test_default_content_surfaces.py::test_default_content_surfaces` | `resources/list` includes `skill://` and `ontology://`; `prompts/list` includes each packaged prompt. |

## Negative and boundary cases

Each proving test above includes the refusal or error case that its row names. A vendor outage or an unauthorised call must return the SDK typed error.

## Quality and release proof

Run `uv run pytest tests` and `pre-commit run --all-files` from the repository root. A planned test is added by the code change that lands the row.
