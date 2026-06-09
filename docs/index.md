# vector-mcp

Retrieval-augmented generation (RAG) **API + MCP Server + Agent** for the
agent-utilities ecosystem — a vendor-neutral retrieval layer over multiple vector
database technologies.

!!! info "Official documentation"
    This site is the canonical reference for `vector-mcp`, maintained alongside every
    release.

[![PyPI](https://img.shields.io/pypi/v/vector-mcp)](https://pypi.org/project/vector-mcp/)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
[![License](https://img.shields.io/pypi/l/vector-mcp)](https://github.com/Knuckles-Team/vector-mcp/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/Knuckles-Team/vector-mcp)

## Overview

`vector-mcp` brings retrieval-augmented generation to AI agents through typed,
deterministic MCP tools backed by a pluggable vector-store layer. A single tool
surface and a single Python API operate over five interchangeable backends —
**ChromaDB**, **PostgreSQL/PGVector**, **Qdrant**, **MongoDB**, and **Couchbase** —
so collections, ingestion, and search behave consistently regardless of the store
behind them. It provides:

- **Action-routed MCP tools** — two consolidated tool modules,
  `vector_collection_management` and `vector_search`, that group every collection and
  retrieval operation to minimize tool bloat in an LLM context.
- **A pluggable backend layer** — semantic, lexical (BM25), and hybrid retrieval
  across ChromaDB, PostgreSQL/PGVector, Qdrant, MongoDB, and Couchbase.
- **An integrated Pydantic-AI graph agent** — a second `vector-agent` server that
  speaks the Agent Control Protocol with an optional web interface.

ChromaDB runs entirely on the local filesystem and needs no external service; the
remaining backends connect to a deployed database — see
[Backing Platform](platform.md).

## Explore the documentation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Installation](installation.md)** — pip, source, backend extras, and the prebuilt Docker image.
- :material-server-network: **[Deployment](deployment.md)** — run the MCP and agent servers, Docker Compose, Caddy + Technitium.
- :material-console: **[Usage](usage.md)** — the MCP tools, the `Api` client, and the agent CLI.
- :material-database-cog: **[Backing Platform](platform.md)** — deploy a vector database with Docker.
- :material-sitemap: **[Overview](overview.md)** — backends, tools, and the standardized package layout.
- :material-tag-multiple: **[Concepts](concepts.md)** — the `CONCEPT:VEC-*` registry.

</div>

## Quick start

```bash
pip install "vector-mcp[chromadb]"
vector-mcp                       # stdio MCP server (default transport)
```

Run it as a network server:

```bash
export EMBEDDING_MODEL_ID=text-embedding-nomic-embed-text-v2-moe
vector-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

See **[Installation](installation.md)** and **[Deployment](deployment.md)** for the
full matrix (PyPI extras, Docker image, all transports, reverse proxy, DNS, and the
agent server).
