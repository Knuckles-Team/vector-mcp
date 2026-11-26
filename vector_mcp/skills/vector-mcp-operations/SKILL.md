---
name: vector-mcp-operations
skill_type: skill
description: >-
  Operate vector-mcp through its governed MCP and Arbora capabilities. Use for
  collection lifecycle, root-confined document ingestion, semantic or lexical
  retrieval, hybrid search, backend readiness, troubleshooting, and sanitized
  verification evidence.
---

# Vector MCP Operations

Use the provider's governed MCP tools through Arbora delegation.

## Workflow

1. Establish the verified GraphSession, tenant, and ACL scope.
2. Discover the installed tool schema. Do not infer stale tool names or
   parameters.
3. Use `medulla` unless the operator has selected and configured another
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

## Steps

1. Discover the installed tool schema with `find`(intent="vector collection") and verify the tool names and parameters you will use are listed.
2. Inspect before changing by calling `vector_list_collections`, and verify the target collection state matches expectations.
3. Run the approved read or write call, such as `vector_search`, `vector_semantic_search`, `vector_lexical_search` or `vector_add_documents`, with bounded `number_results`, and verify the response status.
4. For deletion or overwrite, obtain explicit approval and call `vector_delete_collection` with `confirm=true`. Verify the collection is absent from `vector_list_collections`.
5. Report only sanitized status, counts and opaque references from `vector_list_collections` results, and verify the report contains no credentials, endpoints or local paths.

## Failure recovery

| Failure | Cause | Recovery |
|---------|-------|----------|
| Tool or parameter not found | Stale tool name assumed | Re-run `find` for the operation and use the schema it returns |
| Request denied | Session lacks `kg:read`, `kg:write` or `kg:admin` for the action | Stop and ask the operator to grant the scope, since ACL state is never bypassed |
| Ingestion rejected | Absolute path or endpoint or credential argument supplied | Pass a relative name under the configured document root and remove connection arguments |

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
