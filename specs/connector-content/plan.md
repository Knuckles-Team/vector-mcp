# CONN-VECTOR-MCP-CONNECTOR-CONTENT — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Architecture

`pyproject.toml` declares `vector-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `vector_mcp` from it.
