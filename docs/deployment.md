# Deployment

<!-- BEGIN GENERATED: deployment-options -->
## Deployment Options

`vector-mcp` exposes its MCP server (console script `vector-mcp`) four ways. Pick the row that
matches where the server runs relative to your MCP client, then copy the matching
`mcp_config.json` below. Replace the `<your-…>` placeholders with the values from the **Configuration / Environment Variables** section. Do not copy concrete secret/endpoint values into a
committed config file — inject the `EMBEDDING_MODELS` registry, supported secret references, TLS
profiles, selected backend, and document root from AgentConfig or the deployment environment.

| # | Option | Transport | Where it runs | `mcp_config.json` key |
|---|--------|-----------|---------------|------------------------|
| 1 | stdio | `stdio` | client launches a subprocess | `command` |
| 2 | Streamable-HTTP (local) | `streamable-http` | a local network port | `command` or `url` |
| 3 | Local container / uv | `stdio` or `streamable-http` | Docker / Podman / uv on this host | `command` or `url` |
| 4 | Remote URL | `streamable-http` | a remote host behind Caddy | `url` |

### 1. stdio (local subprocess)

The client launches the server over stdio via `uvx` — best for local IDEs
(Cursor, Claude Desktop, VS Code). A client can launch the package without a checkout:

```json
{
  "mcpServers": {
    "vector-mcp": {
      "command": "uvx",
      "args": ["--from", "vector-mcp[mcp]", "vector-mcp"],
      "env": {
        "DATABASE_TYPE": "epistemic_graph"
      }
    }
  }
}
```

The packaged agent-launch configuration contains only the local command, tool mode, and tool
toggles; it inherits operator-managed runtime settings.

### 2. Streamable-HTTP (local process)

Run the server as a long-lived HTTP process:

```bash
uvx --from vector-mcp vector-mcp --transport streamable-http --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/health        # {"status":"OK"}
```

Then either let the client launch it:

```json
{
  "mcpServers": {
    "vector-mcp": {
      "command": "uvx",
      "args": ["--from", "vector-mcp", "vector-mcp", "--transport", "streamable-http", "--port", "8000"],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "DATABASE_TYPE": "epistemic_graph"
      }
    }
  }
}
```

…or connect to the already-running process by URL:

```json
{
  "mcpServers": {
    "vector-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

### 3. Local container / uv

**(a) Launch a container directly from `mcp_config.json`** (stdio over the container —
no ports to manage). Swap `docker` for `podman` for a daemonless runtime:

```json
{
  "mcpServers": {
    "vector-mcp": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "TRANSPORT=stdio",
        "-e", "DATABASE_TYPE=epistemic_graph",
        "knucklessg1/vector-mcp:latest"
      ]
    }
  }
}
```

**(b) Run a local streamable-http container, then connect by URL:**

```bash
docker run -d --name vector-mcp -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e DATABASE_TYPE=epistemic_graph \
  knucklessg1/vector-mcp:latest
# or, from a clone of this repo:
docker compose -f docker/mcp.compose.yml up -d
```

```json
{
  "mcpServers": {
    "vector-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

**(c) From a local checkout with `uv`:**

```bash
uv run vector-mcp --transport streamable-http --port 8000
```

### 4. Remote URL (deployed behind Caddy)

When the server is deployed remotely (e.g. as a Docker service) and published through
Caddy on the internal `*.arpa` zone, connect with the `"url"` key — no local process or
image required:

```json
{
  "mcpServers": {
    "vector-mcp": { "url": "http://vector-mcp.arpa/mcp" }
  }
}
```

Caddy reverse-proxies `http://vector-mcp.arpa` to the container's `:8000`
streamable-http listener; `http://vector-mcp.arpa/health` returns
`{"status":"OK"}` when the service is live.
<!-- END GENERATED: deployment-options -->

## Managed network service

Run the same command with the deployment's streamable-HTTP transport settings, bind only to the
intended interface, and enforce the shared MCP authentication middleware. Publish it behind a
verified HTTPS endpoint and configure clients with the deployment-provided URL:

```json
{
  "mcpServers": {
    "vector-mcp": {
      "url": "https://vector.example.invalid/mcp"
    }
  }
}
```

A remote service must not use plaintext transport. Health probes should return status only and
must not disclose package versions, provider topology, endpoints, identities, or credentials.

## Containers

The maintained container definitions live under `docker/`. Provide all secrets at runtime and
mount any CA bundle or document root read-only. Offline provider tests mock SDK boundaries;
deployment qualification must provision its own authenticated services.

## Release gates

Before deployment:

1. validate Python, JSON, TOML, YAML, and Markdown structure;
2. run the repository privacy and security contracts;
3. verify the selected provider and TLS profile with `vector-mcp-doctor`;
4. certify the installed MCP tool schema and regenerate signed connector evidence;
5. confirm traces contain status, counts, and opaque references only.
