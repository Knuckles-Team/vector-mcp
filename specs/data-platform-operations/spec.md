# CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS — Operations the data-platform vocabulary needs

Owner: vector-mcp. Cross-repo IDs: `MEDULLA-DATAPLAT-VOCAB-R009`.

## Purpose and user stories

The medulla data-platform vocabulary answers competency questions from connector facts. One question needs an operation this connector lacks. This spec owns that operation.

## Requirements and acceptance

See [requirements.md](requirements.md).

## Why a new spec

The existing specs in `specs/` cover packaged content and identity, not tool operations. None owns a new connector operation.

## Scope and interfaces

Owned: the one operation named in the row, touching `vector_mcp/mcp_server.py`. Non-goal: anything in the medulla vocabulary itself.

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS-R001 | plan.md | T-001 |  |

## Open questions

- None.
