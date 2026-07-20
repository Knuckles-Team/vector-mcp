# Deployment

## Local stdio

A client can launch the package without a checkout:

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

Inject the `EMBEDDING_MODELS` registry, supported secret references, TLS profiles, selected
backend, and document root from AgentConfig or the deployment environment. Do not copy concrete values
into this file. The packaged agent-launch configuration contains only the local command,
tool mode, and tool toggles; it inherits operator-managed runtime settings.

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
