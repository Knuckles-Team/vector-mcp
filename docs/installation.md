# Installation

Install the MCP runtime without optional database SDKs:

```bash
uvx --from 'vector-mcp[mcp]' vector-mcp --help
```

Add only the provider extras used by the deployment:

```bash
uv add 'vector-mcp[postgres]'
uv add 'vector-mcp[qdrant]'
uv add 'vector-mcp[mongodb]'
```

Agent Utilities 1.27.1 or newer supplies the native self-contained
`epistemic-graph[full]` runtime. Partial or numeric-only engine installations are not
supported. Providers that cannot satisfy indexed semantic and lexical retrieval with
runtime-only credentials and verified transport are intentionally absent.

Verify configuration without printing values:

```bash
vector-mcp-doctor
```
