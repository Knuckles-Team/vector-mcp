# vector-mcp

Action-routed MCP and agent interfaces for governed vector collection management and retrieval.
The native default is epistemic-graph. Secure opt-in providers cover PostgreSQL/pgvector,
Qdrant, and MongoDB Atlas.

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

The runtime requires `agent-utilities>=1.27.1` and its self-contained full
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
injected by the operator. A client launch entry can remain equally small:

```json
{
  "mcpServers": {
    "vector-mcp": {
      "command": "uvx",
      "args": ["--from", "vector-mcp[mcp]", "vector-mcp"],
      "env": {"MCP_TOOL_MODE": "condensed"}
    }
  }
}
```

Do not put concrete endpoints, certificate paths, credentials, or user directories in the
repository. Durable credentials and TLS profiles should be supplied by secret reference.

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
