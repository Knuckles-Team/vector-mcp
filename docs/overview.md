# Architecture overview

The current server uses two condensed MCP tools:

```text
MCP client
  -> vector_collection_management / vector_search
  -> bounded validation and privacy controls
  -> authenticated current API boundary
  -> configured vector provider
```

Collection operations and search share a canonical backend-name policy. Epistemic-graph is the
native default. Optional providers are imported only when selected so an unused database SDK does
not expand the default runtime.

Document ingestion is root-confined. An operator configures one document root; callers may select
that root or relative files beneath it. The resolver rejects absolute paths, URLs, traversal,
symbolic links, excessive file counts, and oversized inputs before provider delegation.

The package contributes four extension surfaces through entry points:

- skills;
- prompts;
- the vector retrieval ontology;
- a read-only collection-inventory source preset.

Generated connector certification is release evidence, not hand-maintained source. It must bind
the exact installed tool schema and ontology and must be signed by an authorized runtime key.

## Enterprise Readiness

All agents in the ecosystem inherit enterprise-grade infrastructure from `agent-utilities`:

| Feature | Status | Source |
|:--------|:-------|:-------|
| **JWT/OIDC Authentication** | ✅ Built-in | `agent-utilities[auth]` — Authlib JWKS + API key middleware |
| **OpenTelemetry Instrumentation** | ✅ Built-in | `agent-utilities[logfire]` — OTLP export, FastAPI auto-instrumentation |
| **HashiCorp Vault Integration** | ✅ Built-in | `agent-utilities[vault]` — `secret://`, `env://`, `vault://` URI schemes |
| **Audit Logging** | ✅ Built-in | Append-only compliance trail with 30+ action types (CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox) |
| **Token Usage Analytics** | ✅ Built-in | 4-bucket tracking with budget alerting (CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox) |
| **Prompt Injection Defense** | ✅ Built-in | 25+ pattern scanner + jailbreak taxonomy (CONCEPT:AU-OS.config.secrets-authentication) |
| **Guardrail Engine** | ✅ Built-in | Input/output interception with block/redact/warn (CONCEPT:AU-OS.governance.reactive-multi-axis-budget) |
| **Action Execution Pipeline** | ✅ Built-in | Token, cost, duration, and node transition limits Dry-run / commit / rollback phases (CONCEPT:AU-ORCH.adapter.kg-graph-materialization) |
| **Resource Scheduling** | ✅ Built-in | Priority queuing + preemption limits (CONCEPT:AU-OS.state.cognitive-scheduler-preemption) |
| **Session Concurrency** | ✅ Built-in | Enqueue/reject/interrupt/rollback (CONCEPT:AU-OS.governance.reactive-multi-axis-budget) |

## Concept Registry

This project implements or inherits the following ecosystem concepts:

| Concept ID | Description | Source |
|:-----------|:------------|:-------|
| ECO-4.1 | MCP & Universal Skills | `agent-utilities` (inherited) |
| KG-2.0 | Active Knowledge Graph | `agent-utilities` (inherited) |
| KG-2.8 | **Retrieval Quality Gate** | `agent-utilities` (inherited) |

> 📖 **Full Registry**: See [`agent-utilities/docs/overview.md`](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/overview.md) for the complete 5-Pillar concept index.
