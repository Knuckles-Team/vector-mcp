# VECTOR — Quickstart

The shortest path that shows the served behaviour working.

```text
uv run pytest test_mcp_server.py test_protocol_compliance.py test_vector_search.py tests/test_error_handling.py tests/test_mcp_server.py tests/test_protocol_compliance.py tests/test_store_extras.py tests/test_vector_search.py
```

The command starts the connector in process and calls the served tools through the MCP client. A test marked planned in `test-spec.md` runs once the code change that lands its row adds it.
