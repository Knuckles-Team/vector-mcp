# AGENTS.md

## Scope

This repository provides the action-routed `vector-mcp` server, optional vector provider
adapters, an agent entry point, packaged skills/prompts, a vector ontology, and a read-only
collection-inventory connector preset.

## Current architecture

```text
MCP client
  -> vector_collection_management / vector_search
  -> validation, privacy, and backend policy
  -> verified GraphSession boundary
  -> configured provider
```

Epistemic-graph is the native default. Optional providers are isolated behind extras and imported
only when selected.

## Non-negotiable design rules

- Do not add legacy paths, fallback implementations, deprecated aliases, compatibility shims, or
  backward-compatibility branches. Migrate the current contract directly.
- Do not accept credentials or local database paths as MCP tool arguments.
- Do not persist endpoints, credentials, personal identity, raw content, hostnames, or local
  filesystem paths in source, skills, docs, traces, reports, fixtures, or generated evidence.
- Resolve runtime values through AgentConfig, environment variables, and supported secret
  references.
- Resolve TLS through the shared transport profile. Do not add certificate-verification bypasses.
- Keep document ingestion confined to the configured root and relative caller-selected paths.
- Keep optional dependencies lazy; importing `vector_mcp` must not start services or require
  every provider SDK.
- Never fabricate connector signatures, live test evidence, trace evidence, or safe dependency
  versions.

## Source layout

- `vector_mcp/mcp_server.py`: condensed MCP registration and request boundary
- `vector_mcp/vector_api.py`: verified session, privacy, secret, TLS, and provider boundary
- `vector_mcp/vectordb/`: optional provider adapters
- `vector_mcp/backend_policy.py`: canonical backend and exposure policy
- `vector_mcp/document_inputs.py`: bounded root-confined ingestion inputs
- `vector_mcp/doctor.py`: privacy-safe readiness output
- `vector_mcp/skills/`, `prompts/`, `ontology/`, `connectors/`: packaged extensions
- `tests/`: unit and explicitly configured integration tests
- `docs/`: operator documentation

Deleted root-level debug scripts and obsolete test compose artifacts must remain deleted.
Maintained container definitions live under `docker/`.

## Development rules

- Use `agent-utilities` primitives for MCP construction, AgentConfig, secrets, transport
  security, and observability.
- Keep public tool schemas bounded and explicit.
- Use Pydantic fields/models where they define a public validation boundary.
- Log stable status and exception types, not values or response bodies.
- Require runtime credentials for integration tests; never supply checked-in password defaults.
- Preserve unrelated user work and do not commit caches, databases, traces, logs, build outputs,
  or environment files.
- Keep `pyproject.toml`, module versions, and `uv.lock` synchronized. Do not regenerate the
  lock as a side effect of unrelated work.

## Cheap validation

These checks do not start providers:

```bash
python scripts/security_sanitizer.py
python scripts/security_contract.py --contract .security/security-contract.json validate
python -m compileall -q vector_mcp
```

Also parse JSON/TOML/YAML, run `git diff --check`, and run the configured formatter/linter.
Provider tests, services, lock regeneration, and native compilation are separate serialized gates
and must not be run concurrently on a constrained workstation.

## Connector evidence

The source preset follows the installed action-routed tool schema. Exact tool-schema fingerprints,
connector manifests, and certification files are generated release evidence. Generate them only
after observing the installed MCP schema and resolving an authorized runtime signing key. If
source ontology or tool schemas change, old signatures are invalid and must not be copied forward.
