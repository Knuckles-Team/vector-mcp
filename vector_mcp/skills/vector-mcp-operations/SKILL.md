---
name: vector-mcp-operations
skill_type: skill
description: >-
  Operate vector-mcp through its governed MCP and GraphOS capabilities. Use for
  collection lifecycle, root-confined document ingestion, semantic or lexical
  retrieval, hybrid search, backend readiness, troubleshooting, and sanitized
  verification evidence.
---

# Vector MCP Operations

Use the provider's governed MCP tools through GraphOS delegation.

## Workflow

1. Establish the verified GraphSession, tenant, and ACL scope.
2. Discover the installed action-routed tool schema. Do not infer stale actions or
   parameters.
3. Use `epistemic_graph` unless the operator has selected and configured another
   currently available backend. Connection details and credentials are never tool
   arguments.
4. Inspect before changing. Fence approved mutations as idempotent WorkItems and require
   explicit confirmation for deletion or overwrite.
5. For file ingestion, accept only relative names beneath the configured document root.
   Inline content must have an approved data contract and retention policy.
6. Select semantic search for conceptual similarity, lexical search for exact terms, and
   hybrid search when both signals matter. Keep result counts and fusion parameters
   bounded.
7. Verify the durable result and report only sanitized status, counts, and opaque
   references.

## Safety contract

- Persist only policy-approved, PII-sanitized document content. Never persist
  credentials, endpoints, raw personal identifiers, hostnames, or local paths.
- Require the verified ambient GraphSession: `kg:read` for search, `kg:write` for
  ingestion, and `kg:admin` for collection lifecycle/inventory. Collection storage
  is partitioned by an opaque tenant digest.
- Resolve credentials through supported runtime secret references and TLS through the
  shared AgentConfig transport profile. Never bypass verification.
- Treat unknown ACL, tenant, schema, or tool-contract state as a hard failure.
- Require explicit approval for destructive, externally visible, or irreversible actions.
- Keep runtime traces metadata-only, policy-scoped, and privacy-sanitized.
- Reject absolute document paths, endpoint or credential tool arguments, and uncertified
  connector activation.

## Specialized workflows

Read [the workflow catalog](references/catalog.md) only when the request needs a
provider-specific procedure, parameter map, script, or reference asset.
