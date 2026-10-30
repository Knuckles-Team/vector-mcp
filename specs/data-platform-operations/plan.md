# CONN-VECTOR-MCP-DATA-PLATFORM-OPERATIONS — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Existing system and reuse

The operation joins the existing tool surface in `vector_mcp/mcp_server.py`. It reuses the connector's client and auth.

## Architecture

One tool per API operation. Read-only. The connector returns typed data; medulla maps it.

## Interfaces and data model

A new MCP tool with a typed result. No migration.

## Live integration path

Client call, then the MCP tool in `vector_mcp/mcp_server.py`, then the backend API.

## Quality and release gates

The repo's pre-commit gates and the focused test named in the row.

## Risks, alternatives, and decisions

A backend without the capability returns a typed unsupported result.
