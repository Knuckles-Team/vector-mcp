# Provider workflow catalog

Load only the workflow relevant to the current request.

## Collection management

Use `vector_collection_management` with one of `create_collection`,
`add_documents`, `delete_collection`, or `list_collections`. Backend connections
and credentials are operator configuration, not tool arguments. Collection names
start with a letter and contain only bounded alphanumeric or underscore characters.

For `create_collection`, set `overwrite` only after approval. For
`add_documents`, supply at least one of the configured-root switch, relative document
names, or policy-approved inline text. File inputs are relative to the configured
ingestion root; URLs, absolute names, traversal, and symbolic links are rejected.
Use `confirm` for deletion only after impact review.

## Retrieval

Use `vector_search` with `semantic_search`, `lexical_search`, or `search`. Bound
result counts and validate the collection identifier before delegation.

- `semantic_search`: conceptual similarity, paraphrases, and recall-oriented discovery.
- `lexical_search`: identifiers, exact phrases, and terminology-sensitive retrieval.
- `search`: hybrid fusion when both signals matter; keep semantic/lexical weights between
  zero and one and use a positive, bounded RRF constant.

Return only content authorized by the active tenant, ACL, and retention policy. Do not
copy retrieved content into traces or validation evidence.

## Backend readiness

The native backend is `epistemic_graph`. The only external providers are `postgres`,
`mongodb`, and `qdrant`; each is opt-in and may be selected only when the installed
backend policy reports it available and the operator has configured its endpoint,
secret references, and TLS profile through AgentConfig.
Never pass a database location, host, username, password, token, or certificate setting
through an MCP action.

## Collection inventory ingestion

The packaged source preset reads `list_collections` through the action-routed
management tool. Activate it only when its observed tool-schema fingerprint and
signed release evidence match the installed package. Inventory metadata is the
entire permitted scope; chunks, embeddings, endpoints, credentials, and source
content are not.
