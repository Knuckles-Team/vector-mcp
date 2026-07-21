# Usage

Select the backend and approved document source through AgentConfig or the runtime
environment, then start the MCP server:

```bash
vector-mcp
```

The document root is supplied by the deployment and is required only for filesystem
ingestion. MCP callers never pass an
absolute host path: use `include_configured_directory=true` to ingest that root,
or provide `document_paths` as relative names beneath it. URLs, traversal,
symbolic links, oversized inputs, and paths outside the configured root are
rejected before a backend sees them.

Discover the current tool schema before ingestion or search. GraphOS callers
should use `vector-mcp-operations`, preserve tenant/ACL metadata, and verify
counts and opaque references without returning raw source content in validation
evidence.

Before persistence, inline and file-derived text passes through the shared PII
sanitizer. Reader metadata that can identify a file, directory, URI, URL, or source
location is discarded, and document identifiers are derived from sanitized content.
