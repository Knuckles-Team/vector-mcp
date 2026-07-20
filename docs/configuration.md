# Configuration, trust, and privacy

This page is the operator contract for `vector-mcp`. Package-specific endpoint,
authentication, tool-toggle, and model settings remain documented in the
repository README and the installed command's `--help` output. Runtime values
must be injected by the launcher; they do not belong in source, packaged skill
content, traces, or generated reports.

`DOCUMENT_DIRECTORY` is the operator-owned ingestion root. Public MCP tools
expose only an `include_configured_directory` switch and relative
`document_paths`; they do not accept arbitrary directories, absolute paths, or
URLs. This keeps workstation paths out of tool arguments and observability data.

## Capability configuration

The current capability surface is defined by three packaged artifacts:

- the action-routed MCP tools described in the README and `docs/usage.md`;
- the compact canonical `vector-mcp-operations` skill and its on-demand catalog;
- the read-only collection-inventory source preset and vector ontology.

Treat those artifacts as a unit during release and deployment. Generate a connector
manifest, tool-schema fingerprint, and certification evidence only from the installed
MCP schema with an authorized runtime signing key; none is checked in as reusable
evidence. Use the compact action-routed surface for delegated agents.

## Runtime values and secrets

- Supply service endpoints and non-secret selectors through AgentConfig or the
  runtime environment. Supply credentials and model keys only through supported
  runtime secret references.
- Use non-personal agent aliases and opaque tenant/correlation identifiers.
- Keep developer directories, workstation names, and deployment hostnames out
  of checked-in configuration.
- Bind network transports to an explicitly chosen interface and require the
  deployment's MCP authentication policy before accepting remote traffic.
- Enable optional agent, embedding, evolution, or observability features only
  when their dependencies and backends are configured and healthy.

The checked-in examples use `localhost` for loopback-only development and
`example.invalid` for replaceable network endpoints. Neither value is a
production default.

## TLS trust

Certificate verification is required. For a private certificate authority,
mount a PEM bundle containing the required intermediate and root certificates,
then configure the client environment with `SSL_CERT_FILE` and, for
Requests-compatible clients, `REQUESTS_CA_BUNDLE`. When `uvx` must use the
native platform trust store while resolving packages, set `UV_NATIVE_TLS=true`.

Do not disable verification to work around an incomplete server chain. Keep CA
bundle locations environment-configured and stable for the runtime; never embed
a workstation path or certificate material in MCP configuration.

### Current transport boundary

The action-routed MCP surface does not accept database credentials or local
database paths. Operators configure the selected backend and its connection
material outside tool arguments, through `AgentConfig`, environment variables,
or secret references supported by the deployed service. This keeps secrets and
workstation paths out of delegation payloads and traces.

Embedding endpoints, providers, model identifiers, and credential references are
configured in the canonical AgentConfig `EMBEDDING_MODELS` registry. Vector MCP calls
the shared `create_embedding_model()` factory and does not introduce a second endpoint
or credential namespace. Embedding trust uses `EMBEDDING_TLS_PROFILE` or
`EMBEDDING_TLS_PROFILE_REF`, including runtime-only complete-chain CA bundles and
optional mTLS material. A checked-in boolean SSL-verification bypass is not supported.

### Backend fields

The native `epistemic_graph` provider needs no vector-specific endpoint or credential;
the engine resolver owns autostart and deployment mode. External providers use these
AgentConfig fields:

| Provider | Connection | Credential references | TLS selector |
| --- | --- | --- | --- |
| PostgreSQL | `DB_HOST`, `DB_PORT`, `DBNAME` | `DB_USERNAME_REF`, `DB_PASSWORD_REF` | `POSTGRES_TLS_PROFILE` or `POSTGRES_TLS_PROFILE_REF` |
| MongoDB Atlas | `DBNAME` | `MONGODB_URI_REF` | `MONGODB_TLS_PROFILE` or `MONGODB_TLS_PROFILE_REF` |
| Qdrant | `DB_HOST`, `DB_PORT`, `QDRANT_HTTP_ALLOWED_PRIVATE_HOSTS` | `QDRANT_API_KEY_REF` | `QDRANT_TLS_PROFILE` or `QDRANT_TLS_PROFILE_REF` |

PostgreSQL always uses `verify-full`; MongoDB always enables certificate and hostname
verification; Qdrant is constructed as HTTPS with the resolved shared TLS profile and a
per-request DNS-pinned transport. Private Qdrant hosts must be listed exactly in
`QDRANT_HTTP_ALLOWED_PRIVATE_HOSTS`. Qdrant's pinned transport does not accept
ambient or profile proxies; deploy it with a direct verified route.
Credential-bearing database URIs are accepted only behind the MongoDB runtime secret
reference and are never logged or returned.

`vector-mcp-doctor` reports only configuration booleans and TLS readiness. It
does not print endpoints, hostnames, credentials, secret references, or
certificate locations. Backend-specific drivers remain optional package
providers; their connection configuration belongs to the process that hosts
those drivers, not the MCP request schema.

## Privacy and data governance

The default observability posture is metadata-only. Do not persist prompts,
message bodies, tool inputs/results, document content, raw traces, credentials,
local paths, hostnames, or personal identity unless an approved data contract
explicitly requires it. Keep Langfuse or OTLP content capture disabled unless a
reviewed retention and access policy authorizes it.

Before vector persistence, content is PII-sanitized and reader metadata keys that
can disclose a file, directory, source, URI, or URL are removed. When connector ingestion is enabled, each change must carry tenant, ACL,
classification, retention, provenance, and checkpoint/delta metadata. Reject or
quarantine records that cannot satisfy that contract; never silently widen a
tenant scope. Logs and reports should contain counts, status, and opaque
references only.

Every collection operation also requires the verified ambient `GraphSession`.
Search requires `kg:read`, document ingestion requires `kg:write`, and collection
lifecycle/inventory requires `kg:admin`. Physical collection names are
prefixed with a one-way tenant partition digest; the tenant identifier itself is
not persisted in the provider namespace or returned by the tool.

## Deployment verification

1. Validate the capability bundle and skill metadata against the installed tool
   schemas.
2. Confirm required secrets are present without printing their values.
3. Verify the complete TLS chain with certificate verification enabled.
4. Exercise health/readiness and one least-privilege read operation.
5. Confirm traces arrive under the expected opaque tenant/run identifiers and
   contain no captured content.
6. Record only sanitized pass/fail evidence and version identifiers.
