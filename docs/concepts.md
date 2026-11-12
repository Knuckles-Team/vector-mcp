# Concept Registry — vector-mcp

> **Prefix**: `CONCEPT:VEC-*`
> **Version**: 1.14.0
> **Bridge**: [`CONCEPT:AU-ECO.messaging.native-backend-abstraction`](https://knuckles-team.github.io/meristem/) (Unified Toolkit Ingestion)
> **Version**: 2.1.2
> **Bridge**: [`CONCEPT:AU-ECO.messaging.native-backend-abstraction`](https://github.com/Knuckles-Team/meristem/blob/main/docs/overview.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:VC-OS.governance.vec` | Collection Management | MCP tool domain `collection_management` — Action-routed dynamic tool registration |
| `CONCEPT:VC-OS.governance.vec-2` | Search & Discovery | MCP tool domain `search` — Action-routed dynamic tool registration |

## Cross-Project References (from meristem)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:AU-ECO.messaging.native-backend-abstraction` | Unified Toolkit Ingestion | meristem |
| `CONCEPT:AU-ORCH.adapter.hot-cache-invalidation` | Confidence-Gated Router | meristem |
| `CONCEPT:AU-OS.config.secrets-authentication` | Prompt Injection Defense | meristem |
| `CONCEPT:AU-OS.state.cognitive-scheduler-preemption` | Cognitive Scheduler | meristem |
| `CONCEPT:AU-OS.governance.reactive-multi-axis-budget` | Guardrail Engine | meristem |
| `CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox` | Audit Logging | meristem |
| `CONCEPT:AU-KG.query.object-graph-mapper` | Knowledge Graph Core | meristem |

## Synergy with meristem

This project integrates with `meristem` via `CONCEPT:AU-ECO.messaging.native-backend-abstraction` (Unified Toolkit Ingestion). The `vector_mcp` MCP server registers its tools with the meristem FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all VEC-* concepts.
