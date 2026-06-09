# Deployment

This page covers running `vector-mcp` as a long-lived server: the transports, a
Docker Compose stack, the companion agent server, putting it behind a Caddy reverse
proxy, and giving it a DNS name with Technitium. To provision the **vector database**
it connects to, see [Backing Platform](platform.md).

> `vector-mcp` ships an **MCP server** (console script `vector-mcp`) and an
> integrated **agent server** (console script `vector-agent`). The agent connects to
> the MCP server over HTTP and exposes an optional web interface.

## Run the MCP server

The transport is selected with `--transport` (or the `TRANSPORT` env var):

=== "stdio (default)"

    ```bash
    vector-mcp
    ```
    For IDE / desktop MCP clients that launch the server as a subprocess.

=== "streamable-http"

    ```bash
    vector-mcp --transport streamable-http --host 0.0.0.0 --port 8000
    ```
    A network server with a `/health` endpoint and `/mcp` route.

=== "sse"

    ```bash
    vector-mcp --transport sse --host 0.0.0.0 --port 8000
    ```

Health check (HTTP transports):

```bash
curl -s http://localhost:8000/health        # {"status":"OK"}
```

## Configuration (environment)

`vector-mcp` is configured entirely from the environment. The **required** set:

| Var | Default | Meaning |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for HTTP transports |
| `PORT` | `8000` | Listen port for HTTP transports |
| `TRANSPORT` | `stdio` | `stdio`, `streamable-http`, or `sse` |
| `VECTOR_URL` | `http://localhost:8000` | Backend / embedding service base URL |
| `EMBEDDING_MODEL_ID` | `text-embedding-nomic-embed-text-v2-moe` | Embedding model id |
| `CHUNK_SIZE` | `512` | Document chunk size for ingestion |
| `COLLECTION_MANAGEMENTTOOL` | `True` | Register the collection-management tool |
| `SEARCHTOOL` | `True` | Register the search tool |

Optional governance and telemetry settings — `ENABLE_OTEL`, the
`OTEL_EXPORTER_OTLP_*` exporter credentials, and the Eunomia policy variables
(`EUNOMIA_TYPE`, `EUNOMIA_POLICY_FILE`, `EUNOMIA_REMOTE_URL`) — are documented in
[`.env.example`](https://github.com/Knuckles-Team/vector-mcp/blob/main/.env.example).
Copy it to `.env` and populate only what you use.

## Docker Compose

The repo ships [`docker/mcp.compose.yml`](https://github.com/Knuckles-Team/vector-mcp/blob/main/docker/mcp.compose.yml).
It reads a sibling `.env` and publishes the HTTP server on `:8000`:

```yaml
services:
  vector-mcp-mcp:
    image: knucklessg1/vector-mcp:latest
    container_name: vector-mcp-mcp
    hostname: vector-mcp-mcp
    restart: always
    env_file:
      - ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
cp .env.example .env          # then edit the embedding + backend values
docker compose -f docker/mcp.compose.yml up -d
docker compose -f docker/mcp.compose.yml logs -f
```

## Agent server

`vector-mcp` includes an integrated Pydantic-AI graph agent exposed by the
`vector-agent` console script. It connects to the MCP server over HTTP (`MCP_URL`),
listens on `:9023`, and serves an optional web interface (`ENABLE_WEB_UI`).

Run it directly:

```bash
export MCP_URL=http://localhost:8000/mcp
vector-agent --provider openai --model-id gpt-4o
```

The repo ships [`docker/agent.compose.yml`](https://github.com/Knuckles-Team/vector-mcp/blob/main/docker/agent.compose.yml),
which deploys the MCP server and the agent together and wires the agent at the MCP
server by container name:

```yaml
services:
  vector-mcp-mcp:
    image: knucklessg1/vector-mcp:latest
    hostname: vector-mcp-mcp
    env_file:
      - ../.env
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"

  vector-mcp-agent:
    image: knucklessg1/vector-mcp:latest
    hostname: vector-mcp-agent
    depends_on:
      - vector-mcp-mcp
    env_file:
      - ../.env
    command: [ "vector-agent" ]
    environment:
      - HOST=0.0.0.0
      - PORT=9023
      - MCP_URL=http://vector-mcp-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
    ports:
      - "9023:9023"
```

```bash
docker compose -f docker/agent.compose.yml up -d
```

## Behind a Caddy reverse proxy

Expose the HTTP servers on hostnames with automatic TLS. Add to your `Caddyfile`:

```caddy
# Internal (self-signed) — homelab .arpa zone
vector-mcp.arpa {
    tls internal
    reverse_proxy vector-mcp-mcp:8000
}

vector-agent.arpa {
    tls internal
    reverse_proxy vector-mcp-agent:9023
}
```

```caddy
# Public — automatic Let's Encrypt
vector-mcp.example.com {
    reverse_proxy vector-mcp-mcp:8000
}
```

Reload Caddy:

```bash
docker compose -f services/caddy/compose.yml exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## DNS with Technitium

Point the hostname at the host running Caddy. Via the Technitium API:

```bash
curl -s "http://technitium.arpa:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=vector-mcp.arpa" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=10.0.0.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `vector-mcp.arpa → <caddy-host-ip>` in the Technitium web
console (`http://technitium.arpa:5380`). The ecosystem
[`technitium-dns-mcp`](https://knuckles-team.github.io/technitium-dns-mcp/) automates
this as a tool.

## Register with an MCP client

Add to your client's `mcp_config.json`:

```json
{
  "mcpServers": {
    "vector-mcp": {
      "command": "uv",
      "args": ["run", "vector-mcp"],
      "env": {
        "EMBEDDING_MODEL_ID": "text-embedding-nomic-embed-text-v2-moe",
        "CHUNK_SIZE": "512",
        "VECTOR_URL": "http://localhost:8000"
      }
    }
  }
}
```

For a remote HTTP server, point the client at `http://vector-mcp.arpa/mcp` instead.
