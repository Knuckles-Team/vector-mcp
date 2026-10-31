# CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS-R001 | Contract | A fixture backend | stats action returns the four fields on a fixture collection. |  |

## Negative and boundary cases

A backend without the capability returns a typed unsupported result.

## Quality and release proof

Run the focused test with `uv run pytest` on the new test file.
