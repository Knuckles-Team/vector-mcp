# Installation

`vector-mcp` is a standard Python package and a prebuilt container image. Pick the
path that matches how you want to run it.

## Requirements

- **Python 3.11+**.
- An embedding model endpoint (for example, an OpenAI-compatible server).
- For any backend other than ChromaDB, a reachable vector database — see
  [Backing Platform](platform.md) to deploy one locally.

## From PyPI (recommended)

```bash
pip install vector-mcp
```

### Optional extras

The base install ships ChromaDB. Add the extra for the backend or capability you
need:

| Extra | Install | Pulls in |
|---|---|---|
| `chromadb` | `pip install "vector-mcp[chromadb]"` | ChromaDB store + OpenTelemetry exporters |
| `postgres` | `pip install "vector-mcp[postgres]"` | `psycopg` + PGVector store |
| `qdrant` | `pip install "vector-mcp[qdrant]"` | Qdrant client + FastEmbed store |
| `mongodb` | `pip install "vector-mcp[mongodb]"` | `pymongo` + MongoDB store |
| `couchbase` | `pip install "vector-mcp[couchbase]"` | Couchbase SDK + store |
| `huggingface` | `pip install "vector-mcp[huggingface]"` | local Sentence-Transformers embeddings |
| `agent` | `pip install "vector-mcp[agent]"` | Pydantic-AI agent + Logfire tracing |
| `all` | `pip install "vector-mcp[all]"` | Every backend, embeddings, and the agent |

```bash
# Typical: run the MCP server against PostgreSQL/PGVector
pip install "vector-mcp[postgres]"
```

## From source

```bash
git clone https://github.com/Knuckles-Team/vector-mcp.git
cd vector-mcp
pip install -e ".[all]"          # editable install with every extra
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[all]"
uv run vector-mcp
```

## Prebuilt Docker image

A multi-stage, slim image is published on every release (installs `vector-mcp[all]`,
entrypoint `vector-mcp`):

```bash
docker pull knucklessg1/vector-mcp:latest

docker run --rm -i \
  -e EMBEDDING_MODEL_ID=text-embedding-nomic-embed-text-v2-moe \
  knucklessg1/vector-mcp:latest        # stdio transport (default)
```

For an HTTP server with a published port and the companion agent, see
[Deployment](deployment.md).

## Verify the install

```bash
vector-mcp --help
python -c "import vector_mcp; print(vector_mcp.__version__)"
```

## Next steps

- **[Deployment](deployment.md)** — run it as a long-lived MCP server behind Caddy + DNS.
- **[Usage](usage.md)** — call the tools, the `Api` client, and the agent CLI.
- **[Backing Platform](platform.md)** — deploy ChromaDB, PostgreSQL, Qdrant, MongoDB, or Couchbase.
