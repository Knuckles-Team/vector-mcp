# VECTOR — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

This plan states how `vector-mcp` delivers the rows of `VECTOR`. The live caller is the `vector-mcp` console script. The owner is the `vector_mcp` package. The rows reuse the `scion-sdk` helpers and add no second authority. The plan changes no runtime code.

## Technical Context

- **Language/Version**: Python >=3.12
- **Primary Dependencies**: scion-sdk[mcp]>=0.6.4,<0.7.0; fastmcp; further pins in `pyproject.toml` (scion-sdk[mcp]>=0.6.4,<0.7.0, markdownify>=1.2.3, beautifulsoup4>=4.15.0, ebooklib>=0.20, html2text>=2025.4.15, ipython>=9.15.0)
- **Storage**: N/A, the connector keeps no store of its own
- **Testing**: pytest; each row names the proving test as `path::name`
- **Target Platform**: Linux container and stdio or streamable HTTP MCP clients
- **Project Type**: connector
- **Performance Goals**: N/A, the vendor service sets the latency
- **Constraints**: vendor calls go through the SDK governed client; no connector-local transport
- **Scale/Scope**: 8 active row(s) in this spec

## Constitution Check

- Organization constitution 1.5.0, spec-first change: pass. Every behaviour is a row of this spec with a bound or planned test.
- Repository article R-I, connector scope: pass. The rows stay inside this connector's one external system and reuse the SDK for shared capabilities.

## Project Structure

- `vector_mcp/vector_api.py` (existing)
- `tests/test_protocol_compliance.py` (existing)
- `vector_mcp/mcp_server.py` (existing)
- `tests/test_mcp_server.py` (existing)
- `vector_mcp/vectordb/base.py` (existing)
- `pyproject.toml` (existing)
- `tests/test_store_extras.py` (existing)
- `tests/test_vector_search.py` (new, planned)
- `test_vector_search.py` (new, planned)
- `test_mcp_server.py` (new, planned)
- `tests/test_error_handling.py` (new, planned)
- `test_protocol_compliance.py` (new, planned)

## Phase 0: Research

No unknowns. Every row names its code and test; the caller chain of each row was read from the code.

## Phase 1: Design

### Summary

This plan states how `vector-mcp` delivers the rows of `VECTOR`. The live caller is the `vector-mcp` console script. The owner is the `vector_mcp` package. The rows reuse the `scion-sdk` helpers and add no second authority. The plan changes no runtime code.

### Technical Context

- **Language/Version**: Python >=3.12
- **Primary Dependencies**: scion-sdk[mcp]>=0.6.4,<0.7.0; fastmcp; further pins in `pyproject.toml` (scion-sdk[mcp]>=0.6.4,<0.7.0, markdownify>=1.2.3, beautifulsoup4>=4.15.0, ebooklib>=0.20, html2text>=2025.4.15, ipython>=9.15.0)
- **Storage**: N/A, the connector keeps no store of its own
- **Testing**: pytest; each row names the proving test as `path::name`
- **Target Platform**: Linux container and stdio or streamable HTTP MCP clients
- **Project Type**: connector
- **Performance Goals**: N/A, the vendor service sets the latency
- **Constraints**: vendor calls go through the SDK governed client; no connector-local transport
- **Scale/Scope**: 8 active row(s) in this spec

### Constitution Check

- Organization constitution 1.5.0, spec-first change: pass. Every behaviour is a row of this spec with a bound or planned test.
- Repository article R-I, connector scope: pass. The rows stay inside this connector's one external system and reuse the SDK for shared capabilities.

### Project Structure

- `vector_mcp/vector_api.py` (existing)
- `tests/test_protocol_compliance.py` (existing)
- `vector_mcp/mcp_server.py` (existing)
- `tests/test_mcp_server.py` (existing)
- `vector_mcp/vectordb/base.py` (existing)
- `pyproject.toml` (existing)
- `tests/test_store_extras.py` (existing)
- `tests/test_vector_search.py` (new, planned)
- `test_vector_search.py` (new, planned)
- `test_mcp_server.py` (new, planned)
- `tests/test_error_handling.py` (new, planned)
- `test_protocol_compliance.py` (new, planned)

### Phase 0: Research

No unknowns. Every row names its code and test; the caller chain of each row was read from the code.

### Phase 1: Design

### Summary

This plan states how `vector-mcp` delivers the rows of `VECTOR`. The live caller is the `vector-mcp` console script. The owner is the `vector_mcp` package. The rows reuse the `scion-sdk` helpers and add no second authority. The plan changes no runtime code.

### Technical Context

- **Language/Version**: Python >=3.12
- **Primary Dependencies**: scion-sdk[mcp]>=0.6.4,<0.7.0; fastmcp; further pins in `pyproject.toml` (scion-sdk[mcp]>=0.6.4,<0.7.0, markdownify>=1.2.3, beautifulsoup4>=4.15.0, ebooklib>=0.20, html2text>=2025.4.15, ipython>=9.15.0)
- **Storage**: N/A, the connector keeps no store of its own
- **Testing**: pytest; each row names the proving test as `path::name`
- **Target Platform**: Linux container and stdio or streamable HTTP MCP clients
- **Project Type**: connector
- **Performance Goals**: N/A, the vendor service sets the latency
- **Constraints**: vendor calls go through the SDK governed client; no connector-local transport
- **Scale/Scope**: 8 active row(s) in this spec

### Constitution Check

- Organization constitution 1.5.0, spec-first change: pass. Every behaviour is a row of this spec with a bound or planned test.
- Repository article R-I, connector scope: pass. The rows stay inside this connector's one external system and reuse the SDK for shared capabilities.

### Project Structure

- `tests/test_mcp_server.py` (existing)
- `tests/test_store_extras.py` (existing)
- `test_vector_search.py` (new, planned)
- `test_mcp_server.py` (new, planned)
- `test_protocol_compliance.py` (new, planned)

### Phase 0: Research

No unknowns. Every row names its code and test; the caller chain of each row was read from the code.

### Phase 1: Design

### Summary

This plan states how `vector-mcp` delivers the rows of `VECTOR`. The live caller is the `vector-mcp` console script. The owner is the `vector_mcp` package. The rows reuse the `scion-sdk` helpers and add no second authority. The plan changes no runtime code.

### Technical Context

- **Language/Version**: Python >=3.12
- **Primary Dependencies**: scion-sdk[mcp]>=0.6.4,<0.7.0; fastmcp; further pins in `pyproject.toml` (scion-sdk[mcp]>=0.6.4,<0.7.0, markdownify>=1.2.3, beautifulsoup4>=4.15.0, ebooklib>=0.20, html2text>=2025.4.15, ipython>=9.15.0)
- **Storage**: N/A, the connector keeps no store of its own
- **Testing**: pytest; each row names the proving test as `path::name`
- **Target Platform**: Linux container and stdio or streamable HTTP MCP clients
- **Project Type**: connector
- **Performance Goals**: N/A, the vendor service sets the latency
- **Constraints**: vendor calls go through the SDK governed client; no connector-local transport
- **Scale/Scope**: 8 active row(s) in this spec

### Constitution Check

- Organization constitution 1.5.0, spec-first change: pass. Every behaviour is a row of this spec with a bound or planned test.
- Repository article R-I, connector scope: pass. The rows stay inside this connector's one external system and reuse the SDK for shared capabilities.

### Project Structure

- `tests/test_mcp_server.py` (existing)
- `tests/test_store_extras.py` (existing)
- `test_vector_search.py` (new, planned)
- `test_mcp_server.py` (new, planned)
- `test_protocol_compliance.py` (new, planned)

### Phase 0: Research

No unknowns. Every row names its code and test; the caller chain of each row was read from the code.

### Phase 1: Design

### Existing system and reuse

VectorAPI abstracts provider selection and backend routing; vector_search MCP tool is the served boundary. VectorDBFactory and backend_policy already codify supported providers (medulla, postgresql, qdrant, mongodb). Embedding model shared via create_embedding_model. No new authority introduced.

### Architecture

MCP client → vector_search tool → VectorAPI.semantic_search → selected backend. Dimension validation occurs before provider invocation. Error cases (mismatch, unsupported type) raise typed refusals. Collection guards separate from query bodies.

### Interfaces and data model

vector_search tool accepts action=semantic_search, query text, and db_type parameter (defaults to medulla). Returns ranked results or refusal. Query embedding and index embedding dimensions must match; mismatch raises vector_dimension_mismatch.

### Live integration path

Client → mcp_server.py vector_search tool → VectorAPI.semantic_search → VectorDBFactory selected backend (medulla, postgresql, qdrant, mongodb) → provider semantic_search or engine ANN index → caller results.

### Quality and release gates

Protocol compliance: test_protocol_compliance.py validates schema for all declared profiles. MCP server integration: test_mcp_server.py verifies tool schema and call path. Error handling: test_error_handling.py covers dimension mismatch and unsupported type. Full path: real tool invocation per profile.

### Risks, alternatives, and decisions

Dimension mismatch is a runtime observable; validation at query time prevents silent failures. Collection creation is separate from query scope to preserve existing guards.

### VECTOR-R001

VECTOR-R001.1-6: vector_search(action=semantic_search) → VectorAPI.semantic_search → backend provider semantic_search → embedding dimension validation → query results or typed refusal.

### Phase 2: Task approach

`tasks.md` holds one task per row, producer first. Rows that depend on none run in parallel. A task is checked when its `Spec:` trailer is on the default branch.

### Complexity Tracking

None. The constitution check passed.

### Build contract

- `pre-commit run --all-files` runs ruff check, ruff format, the shared `connector-standard`, `spec`, `secrets` and `dependency-readiness` gates at the pinned shared revision.
- `uv run pytest tests` runs the proving tests; the shared complexity, duplication and KISS gates apply their org thresholds and this repository sets no local override.
- A change adds no scanner suppression and no duplicate implementation of an SDK helper.

### Production caller chain


### Cross-repo seams

None. The rows depend only on the published `scion-sdk` package, which the connector imports.

### Phase 2: Task approach

`tasks.md` holds one task per row, producer first. Rows that depend on none run in parallel. A task is checked when its `Spec:` trailer is on the default branch.

### Complexity Tracking

None. The constitution check passed.

### Build contract

- `pre-commit run --all-files` runs ruff check, ruff format, the shared `connector-standard`, `spec`, `secrets` and `dependency-readiness` gates at the pinned shared revision.
- `uv run pytest tests` runs the proving tests; the shared complexity, duplication and KISS gates apply their org thresholds and this repository sets no local override.
- A change adds no scanner suppression and no duplicate implementation of an SDK helper.

### Production caller chain


### Cross-repo seams

None. The rows depend only on the published `scion-sdk` package, which the connector imports.

### Phase 2: Task approach

`tasks.md` holds one task per row, producer first. Rows that depend on none run in parallel. A task is checked when its `Spec:` trailer is on the default branch.

### Complexity Tracking

None. The constitution check passed.

### Build contract

- `pre-commit run --all-files` runs ruff check, ruff format, the shared `connector-standard`, `spec`, `secrets` and `dependency-readiness` gates at the pinned shared revision.
- `uv run pytest tests` runs the proving tests; the shared complexity, duplication and KISS gates apply their org thresholds and this repository sets no local override.
- A change adds no scanner suppression and no duplicate implementation of an SDK helper.

### Production caller chain

- `VECTOR-R001.1`: `vector_collection_management` (vector_mcp/mcp_server.py:176) creates the collection through `VectorAPI` in `vector_mcp/vector_api.py`, which asks the selected backend of `vector_mcp/vectordb/` for a schema sized to the embedding dimension.
- `VECTOR-R001.2`: `vector_search` (vector_mcp/mcp_server.py:332) with action `semantic_search` calls `VectorAPI.semantic_search` (vector_mcp/vector_api.py:505), which calls the backend `semantic_search`.
- `VECTOR-R001.3`: `vector_search` (vector_mcp/mcp_server.py:332) gets the client through `get_client`, which builds the backend with the database factory of `vector_mcp/vectordb/`.
- `VECTOR-R001.4`: `vector_search` and `vector_collection_management` call `VectorAPI`, which raises `vector_dimension_mismatch` on a size mismatch; the backend factory raises `vector_database_type_unsupported` (vector_mcp/vectordb/base.py:322).
- `VECTOR-R001.5`: `vector_search` and `vector_collection_management` reach each supported provider profile through the backend factory of `vector_mcp/vectordb/`.
- `VECTOR-R001.6`: an MCP client calls `vector_search` (vector_mcp/mcp_server.py:332) with action `semantic_search`, which reaches `VectorAPI.semantic_search`.
- `VECTOR-R001.7`: the installer reads the optional-dependency extras `postgres`, `qdrant`, `mongodb` and `chromadb` in `pyproject.toml`.

### Cross-repo seams

None. The rows depend only on the published `scion-sdk` package, which the connector imports.

## Phase 2: Task approach

`tasks.md` holds one task per row, producer first. Rows that depend on none run in parallel. A task is checked when its `Spec:` trailer is on the default branch.

## Complexity Tracking

None. The constitution check passed.

## Build contract

- `pre-commit run --all-files` runs ruff check, ruff format, the shared `connector-standard`, `spec`, `secrets` and `dependency-readiness` gates at the pinned shared revision.
- `uv run pytest tests` runs the proving tests; the shared complexity, duplication and KISS gates apply their org thresholds and this repository sets no local override.
- A change adds no scanner suppression and no duplicate implementation of an SDK helper.

## Production caller chain

- `VECTOR-R001.1`: `vector_collection_management` (vector_mcp/mcp_server.py:176) creates the collection through `VectorAPI` in `vector_mcp/vector_api.py`, which asks the selected backend of `vector_mcp/vectordb/` for a schema sized to the embedding dimension.
- `VECTOR-R001.2`: `vector_search` (vector_mcp/mcp_server.py:332) with action `semantic_search` calls `VectorAPI.semantic_search` (vector_mcp/vector_api.py:505), which calls the backend `semantic_search`.
- `VECTOR-R001.3`: `vector_search` (vector_mcp/mcp_server.py:332) gets the client through `get_client`, which builds the backend with the database factory of `vector_mcp/vectordb/`.
- `VECTOR-R001.4`: `vector_search` and `vector_collection_management` call `VectorAPI`, which raises `vector_dimension_mismatch` on a size mismatch; the backend factory raises `vector_database_type_unsupported` (vector_mcp/vectordb/base.py:322).
- `VECTOR-R001.5`: `vector_search` and `vector_collection_management` reach each supported provider profile through the backend factory of `vector_mcp/vectordb/`.
- `VECTOR-R001.6`: an MCP client calls `vector_search` (vector_mcp/mcp_server.py:332) with action `semantic_search`, which reaches `VectorAPI.semantic_search`.
- `VECTOR-R001.7`: the installer reads the optional-dependency extras `postgres`, `qdrant`, `mongodb` and `chromadb` in `pyproject.toml`.

## Cross-repo seams

None. The rows depend only on the published `scion-sdk` package, which the connector imports.
