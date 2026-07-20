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
