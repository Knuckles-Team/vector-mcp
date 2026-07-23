# vector-mcp

Action-routed MCP and agent interfaces for governed vector collection management and retrieval.
The native default is epistemic-graph. Secure opt-in providers cover PostgreSQL/pgvector,
Qdrant, and MongoDB Atlas.

*Version: 3.0.0*

<!-- GOVERNED-CAPABILITY:START -->
## Governed capability

- MCP tools: `vector_collection_management` and `vector_search`
- Skill provider: the consolidated `vector-mcp-operations` workflow
- Ontology provider: the packaged vector retrieval ontology
- Source connector provider: a read-only vector collection inventory preset
- Runtime configuration: AgentConfig, environment variables, and secret references
- Privacy posture: no checked-in endpoints, credentials, personal identity, or host paths
<!-- GOVERNED-CAPABILITY:END -->

## Install

Use the smallest extra set required by the deployment:

```bash
uvx --from 'vector-mcp[mcp]' vector-mcp
```

The runtime requires `agent-utilities>=2.0.0` and its self-contained full
epistemic-graph engine contract. A bare numeric-only or partial engine profile is not a
supported deployment.

For a selected storage provider:

```bash
uv add 'vector-mcp[postgres]'
uv add 'vector-mcp[qdrant]'
uv add 'vector-mcp[mongodb]'
```

The `all` extra enables every supported optional provider plus the agent, Langfuse, and
Logfire runtimes. Production images should install only the providers they operate.

## MCP configuration

The package includes a neutral agent-launch configuration containing only the command,
condensed tool mode, and tool toggles. Runtime values are inherited from AgentConfig or
injected by the operator. Detailed instructions on how to use the underlying API wrappers,
extended schema bindings, and developer SDK references are maintained in
[docs/index.md](docs/index.md).

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools

_Auto-generated from the live MCP server — do not edit by hand._

<!-- MCP-TOOLS-TABLE:START -->

#### Condensed action-routed tools (default — `MCP_TOOL_MODE=condensed`)

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `vector_collection_management` | `COLLECTION_MANAGEMENTTOOL` | Manage collection management operations. |

#### Verbose 1:1 API-mapped tools (`MCP_TOOL_MODE=verbose` or `both`)

<details>
<summary>7 per-operation tools — one per public API method (click to expand)</summary>

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `vector_add_documents` | `APITOOL` | Add documents. |
| `vector_create_collection` | `APITOOL` | Create a collection. |
| `vector_delete_collection` | `APITOOL` | Delete a collection. |
| `vector_lexical_search` | `APITOOL` | Perform lexical search. |
| `vector_list_collections` | `APITOOL` | List collections. |
| `vector_search` | `SEARCHTOOL` | Perform hybrid search. |
| `vector_semantic_search` | `APITOOL` | Perform semantic search. |

</details>

_1 action-routed tool(s) (default) · 7 verbose 1:1 tool(s). Each is enabled unless its `<DOMAIN>TOOL` toggle is set false; `MCP_TOOL_MODE` selects the surface (`condensed` default · `verbose` 1:1 · `both`). Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/mcp.md](docs/mcp.md).

### Dynamic Tool Selection & Visibility

This MCP server supports dynamic toolset selection and visibility filtering at runtime. This allows you to restrict the set of exposed tools in order to prevent blowing up the LLM's context window.

You can configure tool filtering via multiple input channels:

- **CLI Arguments:** Pass `--tools` or `--toolsets` (or their disabled counterparts `--disabled-tools` and `--disabled-toolsets`) during startup.
- **Environment Variables:** Define standard environment variables:
  - `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS`
  - `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS`
- **HTTP SSE Request Headers:** Pass custom headers during transport initialization:
  - `x-mcp-enabled-tools` / `x-mcp-disabled-tools`
  - `x-mcp-enabled-tags` / `x-mcp-disabled-tags`
- **HTTP SSE Request Query Parameters:** Append query parameters directly to your transport connection URL:
  - `?tools=tool1,tool2`
  - `?tags=tag1`

When query strings or parameters are supplied, an LLM-free **Knowledge Graph resolution layer** (using `DynamicToolOrchestrator`) matches query intents against known tool tags, names, or descriptions, with safe fallback and automated 24-hour background cache refreshing.

---

### MCP Configuration Examples

<!-- MCP-CONFIG-EXAMPLES:START -->

> **Install the slim `[mcp]` extra.** All examples install `vector-mcp[mcp]` — the
> MCP-server extra that pulls only the FastMCP / FastAPI tooling (`agent-utilities[mcp]`).
> It deliberately **excludes** the heavy agent runtime (`pydantic-ai`, the epistemic-graph
> engine, `dspy`, `llama-index`), so `uvx` / container installs are far smaller. Use the
> full `[agent]` extra only when you need the integrated Pydantic AI agent.

#### stdio Transport (local IDEs — Cursor, Claude Desktop, VS Code)

A client launch entry can remain equally small:

```json
{
  "mcpServers": {
    "vector-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "vector-mcp[mcp]",
        "vector-mcp"
      ],
      "env": {
        "MCP_TOOL_MODE": "condensed",
        "COLLECTION_MANAGEMENTTOOL": "True",
        "SEARCHTOOL": "True",
        "DATABASE_TYPE": "epistemic_graph"
      }
    }
  }
}
```

#### Streamable-HTTP Transport (networked / production)

```json
{
  "mcpServers": {
    "vector-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "vector-mcp[mcp]",
        "vector-mcp",
        "--transport",
        "streamable-http",
        "--port",
        "8000"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "MCP_TOOL_MODE": "condensed",
        "COLLECTION_MANAGEMENTTOOL": "True",
        "SEARCHTOOL": "True",
        "DATABASE_TYPE": "epistemic_graph"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed Streamable-HTTP instance by `url`. Do not put concrete
endpoints, certificate paths, credentials, or user directories in the repository. Durable
credentials and TLS profiles should be supplied by secret reference.

## Tool surface

### `vector_collection_management`

Supported actions:

- `create_collection`
- `add_documents`
- `delete_collection`
- `list_collections`

Database credentials and local database paths are not accepted as tool arguments. Document
files are selected only by paths relative to the administrator-owned `DOCUMENT_DIRECTORY`.
Absolute paths, URLs, traversal, symbolic links, unbounded file sets, and oversized content are
rejected before delegation.

### `vector_search`

Supported actions:

- `semantic_search`
- `lexical_search`
- `search` (hybrid retrieval)

Backend names, collection identifiers, result counts, search weights, and request sizes are
validated at the MCP boundary. Legacy aliases and providers without the common indexed,
authenticated, verified-TLS contract are not advertised or accepted.

## Runtime trust

Embedding providers are selected through the shared AgentConfig `EMBEDDING_MODELS` registry.
Model credentials remain references in that registry, and model trust is selected through
`EMBEDDING_TLS_PROFILE` or `EMBEDDING_TLS_PROFILE_REF`. Database credentials likewise use only
`env://`, `vault://`, or `secret://` references. Complete-chain PEM bundles, system trust, mTLS,
and proxy policy are resolved by Agent Utilities at runtime; boolean certificate-verification
bypasses are not supported. The doctor reports configuration booleans and
readiness without printing endpoints, hostnames, paths, identities, or secret references:

```bash
vector-mcp-doctor
```

_Auto-generated from the code-read env surface (`MCP_TOOL_MODE` + package vars) — do not edit._
<!-- MCP-CONFIG-EXAMPLES:END -->

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`vector-mcp` can also run as a **local container** (Docker / Podman / `uv`) or be
consumed from a **remote deployment**. The
[Deployment guide](https://knuckles-team.github.io/vector-mcp/deployment/) has full, copy-paste
`mcp_config.json` for all four transports — **stdio**, **streamable-http**,
**local container / uv**, and **remote URL**:

- **Local container / uv** — launch the server from `mcp_config.json` via `uvx`,
  `docker run`, or `podman run`, or point at a local streamable-http container by `url`.
- **Remote URL** — connect to a server deployed behind Caddy at
  `http://vector-mcp.arpa/mcp` using the `"url"` key.
<!-- END GENERATED: additional-deployment-options -->

---

## Environment Variables

<!-- ENV-VARS-TABLE:START -->

#### Package environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` |  |
| `PORT` | `8000` |  |
| `TRANSPORT` | `stdio` | options: stdio, streamable-http, sse |
| `ENABLE_OTEL` | `True` |  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:8080/api/public/otel` |  |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` | `pk-...` |  |
| `OTEL_EXPORTER_OTLP_SECRET_KEY` | `sk-...` |  |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |  |
| `EUNOMIA_TYPE` | `none` | options: none, embedded, remote |
| `EUNOMIA_POLICY_FILE` | `mcp_policies.json` |  |
| `EUNOMIA_REMOTE_URL` | `http://eunomia-server:8000` |  |
| `LLM_BASE_URL` | `http://localhost:8000/v1` | embedding/LLM API base url |
| `LLM_TOKEN` | — | bearer token for the embedding/LLM endpoint |
| `LLM_API_KEY` | — | alias accepted if LLM_TOKEN is unset |
| `LLM_SSL_VERIFY` | `False` | verify TLS for the embedding/LLM endpoint |
| `DOCUMENT_DIRECTORY` | `/documents` | default directory for ingested documents |
| `COLLECTION_MANAGEMENTTOOL` | `True` |  |
| `SEARCHTOOL` | `True` |  |
| `TEST_POSTGRES_CONNECTION_STRING` | `postgresql://postgres:password@localhost:5432/vectordb` |  |
| `TEST_MONGODB_HOST` | `localhost` |  |
| `TEST_MONGODB_PORT` | `27017` |  |
| `TEST_MONGODB_DB` | `vectordb` |  |
| `TEST_QDRANT_LOCATION` | `http://localhost:6333` |  |
| `TEST_COUCHBASE_CONNECTION` | `couchbase://localhost` |  |
| `TEST_COUCHBASE_USER` | `Administrator` |  |
| `TEST_COUCHBASE_PASSWORD` | `password` |  |
| `TEST_COUCHBASE_DB` | `vector_db` |  |

#### Inherited agent-utilities variables (apply to every connector)

| Variable | Example | Description |
|----------|---------|-------------|
| `MCP_TOOL_MODE` | `condensed` | Tool surface: `condensed` | `verbose` | `both` |
| `MCP_ENABLED_TOOLS` | — | Comma-separated tool allow-list |
| `MCP_DISABLED_TOOLS` | — | Comma-separated tool deny-list |
| `MCP_ENABLED_TAGS` | — | Comma-separated tag allow-list |
| `MCP_DISABLED_TAGS` | — | Comma-separated tag deny-list |
| `MCP_CLIENT_AUTH` | — | Outbound MCP auth (`oidc-client-credentials` for fleet calls) |
| `OIDC_CLIENT_ID` | — | OIDC client id (service-account auth) |
| `OIDC_CLIENT_SECRET` | — | OIDC client secret (service-account auth) |
| `DEBUG` | `False` | Verbose logging |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout (recommended in containers) |
| `MCP_URL` | `http://localhost:8000/mcp` | URL of the MCP server the agent connects to |
| `PROVIDER` | `openai` | LLM provider for the agent |
| `MODEL_ID` | `gpt-4o` | Model id for the agent |
| `ENABLE_WEB_UI` | `True` | Serve the AG-UI web interface |

_27 package + 14 inherited variable(s). Auto-generated from `.env.example` + the shared agent-utilities set — do not edit._
<!-- ENV-VARS-TABLE:END -->


Every variable the server reads, grouped by purpose. See [`.env.example`](.env.example) for the
canonical, copy-paste list — including the `DATABASE_TYPE` / `GRAPH_SERVICE_SOCKET` /
`GRAPH_SERVICE_AUTH_SECRET` connection settings for the native epistemic-graph backend. Backend
endpoints, database locations, and credentials for opt-in providers (Postgres/Qdrant/Mongo/
Chroma/Couchbase) are never README-documented literal values or MCP tool arguments — they resolve
through AgentConfig and `secret://`/`env://`/`vault://` references at runtime.

### MCP server / transport
| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSPORT` | `stdio`, `streamable-http`, or `sse` | `stdio` |
| `HOST` | Bind host (HTTP transports) | `0.0.0.0` |
| `PORT` | Bind port (HTTP transports) | `8000` |
| `MCP_TOOL_MODE` | Tool surface: `condensed`, `verbose`, or `both` | `condensed` |
| `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS` | Comma-separated tool allow/deny list | — |
| `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS` | Comma-separated tag allow/deny list | — |
| `PYTHONUNBUFFERED` | Unbuffered stdout (recommended in containers) | `1` |

### Tool toggles
Each action-routed tool can be disabled individually via its toggle env var (set to `false`).
The full list is in the [Available MCP Tools](#available-mcp-tools) table above.

| Variable | Description | Default |
|----------|-------------|---------|
| `COLLECTION_MANAGEMENTTOOL` | Enable the collection-management tool | `True` |
| `SEARCHTOOL` | Enable the search tool | `True` |

### Telemetry & governance
| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_OTEL` | Enable OpenTelemetry export | `True` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | — |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` / `OTEL_EXPORTER_OTLP_SECRET_KEY` | OTLP auth keys | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol (e.g. `http/protobuf`) | — |
| `EUNOMIA_TYPE` | Authorization mode: `none`, `embedded`, `remote` | `none` |
| `EUNOMIA_POLICY_FILE` | Embedded policy file | `mcp_policies.json` |
| `EUNOMIA_REMOTE_URL` | Remote Eunomia server URL | — |

### Agent CLI (full `[agent]` runtime only)
| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_URL` | URL of the MCP server the agent connects to | `http://localhost:8000/mcp` |
| `PROVIDER` | LLM provider (e.g. `openai`) | `openai` |
| `MODEL_ID` | Model id (e.g. `gpt-4o`) | `gpt-4o` |
| `ENABLE_WEB_UI` | Serve the AG-UI web interface | `True` |

See [`.env.example`](.env.example) for a copy-paste starting point.

## Provider and ontology integration

The package contributes its skills, prompts, ontology, and source connector through Python entry
points. The collection-inventory connector is intentionally read-only and registers collection
metadata, not document or embedding payloads.

Generated connector signatures must be recreated only after the installed MCP schema is observed
and a release signing key is provided at runtime. A signature from an older tool schema or
ontology must never be copied forward.

## Development checks

Low-cost checks that do not launch providers:

```bash
python scripts/security_sanitizer.py
python scripts/security_contract.py --contract .security/security-contract.json validate
python -m compileall -q vector_mcp
```

Provider tests use mocked SDK boundaries and make no network calls. Live qualification is a
separate deployment gate and must use operator-supplied AgentConfig and secrets.

## Documentation

- [Installation](docs/installation.md)
- [Configuration and privacy](docs/configuration.md)
- [Deployment](docs/deployment.md)
- [Usage](docs/usage.md)
- [Architecture overview](docs/overview.md)

The slim `:mcp` streamable-http container (`docker/mcp.compose.yml`) publishes `:8000` with a
`/health` check; see [Deployment](docs/deployment.md) for the full compose service definition.

## License

See [LICENSE](LICENSE).


<!-- BEGIN agent-utilities-deployment (generated; do not edit between markers) -->

## Deploy with `agent-utilities-deployment`

Provision this package with the consolidated **`agent-utilities-deployment`**
workflow. It selects an installed-package, editable-source, or immutable-container
path; records only runtime secret and TLS-profile references in `AgentConfig`; and
runs doctor, registration, policy, observability, and rollback gates. Ask your agent
to **"deploy `vector-mcp` with agent-utilities-deployment"**.

| Install mode | Command |
|------|---------|
| Installed package | `uv tool install "vector-mcp[mcp]"`, then run `vector-mcp` |
| Editable source | `uv pip install -e ".[agent]"`, then run `vector-mcp` |
| Immutable container | deploy `registry.example.invalid/vector-mcp@sha256:<digest>` through the operator-selected orchestrator |

The repository embeds no deployment profile, credential value, certificate path, or
environment-specific endpoint. Supply those at runtime through `AgentConfig` and the
configured secret provider.

<!-- END agent-utilities-deployment -->

---

## Installation

Pick the extra that matches what you want to run:

| Extra | Installs | Use when |
|-------|----------|----------|
| `vector-mcp[mcp]` | Slim MCP server only (`agent-utilities[mcp]` — FastMCP/FastAPI) | You only run the **MCP server** (smallest install / image) |
| `vector-mcp[agent]` | Full agent runtime (`agent-utilities[agent,logfire]` — Pydantic AI + the epistemic-graph engine) | You run the **integrated agent** |
| `vector-mcp[all]` | Everything (`mcp` + all vector backends + `agent`) | Development / both surfaces |

```bash
# MCP server only (recommended for tool hosting — slim deps)
uv pip install "vector-mcp[mcp]"

# Full agent runtime (Pydantic AI + epistemic-graph engine)
uv pip install "vector-mcp[agent]"

# Everything (development)
uv pip install "vector-mcp[all]"      # or: python -m pip install "vector-mcp[all]"
```

### Container images (`:mcp` vs `:agent`)

One multi-stage `docker/Dockerfile` builds two right-sized images, selected by `--target`:

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `knucklessg1/vector-mcp:mcp` | `--target mcp` | `vector-mcp[mcp]` — **slim**, no engine/`pydantic-ai`/`dspy`/`llama-index`/`tree-sitter` | `vector-mcp` |
| `knucklessg1/vector-mcp:latest` | `--target agent` (default) | `vector-mcp[agent]` — **full** agent runtime + epistemic-graph engine | `vector-agent` |

```bash
docker build --target mcp   -t knucklessg1/vector-mcp:mcp    docker/   # slim MCP server
docker build --target agent -t knucklessg1/vector-mcp:latest docker/   # full agent
```

`docker/mcp.compose.yml` runs the slim `:mcp` server; `docker/agent.compose.yml` runs the
agent (`:latest`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

The **full agent** (`[agent]` / `:latest`) embeds the **epistemic-graph** engine (pulled in
transitively via `agent-utilities[agent]`). For production — or to share one knowledge graph
across multiple agents — run **epistemic-graph as its own database container** and point the
agent at it instead of embedding it. Deployment recipes (single-node + Raft HA), connection
config, and the full database architecture (with diagrams) are documented in the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).
The slim `[mcp]` server does **not** require the database.

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)

---

## Contribute

Contributions are welcome! Please ensure code quality by executing local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Execute test suites using `pytest`


<!-- BEGIN agent-os-genesis-deploy (generated; do not edit between markers) -->

## Deploy with `agent-os-genesis`

This package can be provisioned for you — skill-guided — by the **`agent-os-genesis`**
universal skill (its *single-package deploy mode*): it picks your install method, seeds
secrets to OpenBao/Vault (or `.env`), trusts your enterprise CA, registers the MCP
server, and verifies it — the same machinery that stands up the whole Agent OS, narrowed
to just this package. Ask your agent to **"deploy `vector-mcp` with agent-os-genesis"**.

| Install mode | Command |
|------|---------|
| Bare-metal, prod (PyPI) | `uvx vector-mcp` · or `uv tool install vector-mcp` |
| Bare-metal, dev (editable) | `uv pip install -e ".[all]"` · or `pip install -e ".[all]"` |
| Container, prod | deploy `knucklessg1/vector-mcp:latest` via docker-compose / swarm / podman / podman-compose / kubernetes |
| Container, dev (editable) | deploy `docker/compose.dev.yml` (source-mounted at `/src`; edits live on restart) |

Secrets are read-existing + seeded via `vault_sync` — you are only prompted for what's missing.

<!-- END agent-os-genesis-deploy -->
