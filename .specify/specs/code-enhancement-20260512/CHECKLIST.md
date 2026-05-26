# Verification Checklist: Code Enhancement: vector-mcp

## Functional Requirements Verification
- [ ] **FR-001**: Minor update: pypdf 6.10.2 (installed) -> 6.11.0
- [ ] **FR-002**: Minor update: llama-index-vector-stores-postgres 0.7.3 (constraint — not installed) -> 0.8.1
- [ ] **FR-003**: Minor update: opentelemetry-api 1.39.1 (installed) -> 1.41.1
- [ ] **FR-004**: MAJOR update: protobuf 6.33.6 (installed) -> 7.34.1
- [ ] **FR-005**: Minor update: opentelemetry-sdk 1.39.1 (installed) -> 1.41.1
- [ ] **FR-006**: Minor update: opentelemetry-exporter-otlp 1.39.1 (constraint — not installed) -> 1.41.1
- [ ] **FR-007**: Minor update: llama-index-vector-stores-couchbase 0.6.0 (constraint — not installed) -> 0.7.1
- [ ] **FR-008**: Minor update: fastembed 0.7.4 (constraint — not installed) -> 0.8.0
- [ ] **FR-009**: Minor update: qdrant-client 1.16.2 (constraint — not installed) -> 1.18.0
- [ ] **FR-010**: Minor update: couchbase 4.5.0 (constraint — not installed) -> 4.6.1
- [ ] **FR-011**: Minor update: llama-index-vector-stores-qdrant 0.9.1 (constraint — not installed) -> 0.10.1
- [ ] **FR-012**: Minor update: llama-index-embeddings-huggingface 0.6.1 (constraint — not installed) -> 0.7.0
- [ ] **FR-013**: Minor update: llama-index-vector-stores-mongodb 0.9.1 (constraint — not installed) -> 0.10.1
- [ ] **FR-014**: Minor update: sentence_transformers 5.2.2 (constraint — not installed) -> 5.5.0
- [ ] **FR-015**: 2 functions exceed 200 lines (actionable refactoring targets): register_collection_management_tools (448L), register_search_tools (337L)
- [ ] **FR-016**: Monolithic: mcp_server.py (1054L) — 3 functions with high complexity (worst: register_collection_management_tools at 448L, CC=39); Low cohesion: 13 distinct concepts in one file
- [ ] **FR-017**: Monolithic: db_utils.py (556L) — 2 functions with high complexity (worst: ModuleInfo.from_str at 52L, CC=10); Low cohesion: 21 distinct concepts in one file
- [ ] **FR-018**: Monolithic: couchbase.py (708L) — 2 functions with high complexity (worst: CouchbaseVectorDB._create_collection_via_rest at 51L, CC=12); God class: CouchbaseVectorDB (23 methods) — consider mixins/composition
- [ ] **FR-019**: 18 functions with nesting depth >4
- [ ] **FR-020**: 8 tests without assertions
- [ ] **FR-021**: 20 potential doc-test drift items
- [ ] **FR-022**: README.md missing sections: installation
- [ ] **FR-023**: README missing: Has a Table of Contents
- [ ] **FR-024**: README missing: References /docs directory material
- [ ] **FR-025**: SRP: 4 modules exceed 500 lines (god modules)
- [ ] **FR-026**: SRP: 1 classes have >15 methods
- [ ] **FR-027**: No discernible layer architecture (no domain/service/adapter separation)
- [ ] **FR-028**: Low traceability ratio: 0% concepts fully traced
- [ ] **FR-029**: 30 test functions missing concept markers
- [ ] **FR-030**: 121 significant functions (>10 lines) missing concept markers in docstrings
- [ ] **FR-031**: Total lint findings: 35 (high/error: 34, medium/warning: 1, low: 0)
- [ ] **FR-032**: 1 hook(s) may be outdated: ruff-pre-commit
- [ ] **FR-033**: 6 rogue/throwaway scripts detected (fix_*, validate_*, patch_*, etc.): scripts/debug_full.py, scripts/validate_all_dbs.py, scripts/debug_embedding.py, scripts/debug_pg.py, scripts/validate_agents.py
- [ ] **FR-034**: CHANGELOG.md is missing — create one following Keep a Changelog format
- [ ] **FR-035**: CHANGELOG.md is missing
- [ ] **FR-036**: 1 test files exceed 500 lines — split into focused modules
- [ ] **FR-037**: Test directory lacks subdirectory organization (consider unit/, integration/, e2e/)
- [ ] **FR-038**: Missing conftest.py for shared fixtures
- [ ] **FR-039**: No shared fixtures in conftest.py
- [ ] **FR-040**: 8 tests have no assertions
- [ ] **FR-041**: Partial env var documentation: 51% coverage
- [ ] **FR-042**: Undocumented env vars: EMBEDDING_MODEL, EMBEDDING_MODEL_ID, EMBEDDING_PROVIDER, ENABLE_OTEL, EUNOMIA_REMOTE_URL, OAUTH_BASE_URL, OAUTH_UPSTREAM_AUTH_ENDPOINT, OAUTH_UPSTREAM_CLIENT_ID, OAUTH_UPSTREAM_CLIENT_SECRET, OAUTH_UPSTREAM_TOKEN_ENDPOINT
- [ ] **FR-043**: 23 Python env vars not in .env.example: API_TOKEN, COLLECTION_MANAGEMENTTOOL, COLLECTION_NAME, DATABASE_PATH, DATABASE_TYPE

## User Stories / Acceptance Criteria
- [ ] As a **developer**, I want to **address Dependency Audit findings (grade: F, score: 50)**, so that **improve project dependency audit from F to at least B (80+)**.
- [ ] As a **developer**, I want to **address Codebase Optimization findings (grade: F, score: 56)**, so that **improve project codebase optimization from F to at least B (80+)**.
- [ ] As a **developer**, I want to **address Test Coverage findings (grade: C, score: 70)**, so that **improve project test coverage from C to at least B (80+)**.
- [ ] As a **developer**, I want to **address Documentation & Governance findings (grade: C, score: 76)**, so that **improve project documentation & governance from C to at least B (80+)**.
- [ ] As a **developer**, I want to **address Architecture & Design Patterns findings (grade: C, score: 70)**, so that **improve project architecture & design patterns from C to at least B (80+)**.
- [ ] As a **developer**, I want to **address Concept Traceability findings (grade: F, score: 30)**, so that **improve project concept traceability from F to at least B (80+)**.
- [ ] As a **developer**, I want to **address Linting & Formatting findings (grade: F, score: 0)**, so that **improve project linting & formatting from F to at least B (80+)**.
- [ ] As a **developer**, I want to **address Changelog Audit findings (grade: C, score: 75)**, so that **improve project changelog audit from C to at least B (80+)**.
- [ ] As a **developer**, I want to **address Pytest Quality findings (grade: C, score: 78)**, so that **improve project pytest quality from C to at least B (80+)**.
- [ ] As a **developer**, I want to **address Environment Variables findings (grade: C, score: 75)**, so that **improve project environment variables from C to at least B (80+)**.

## Success Criteria
- [ ] Overall GPA: 2.24 → 3.0
- [ ] Domains at B or above: 7 → 17
- [ ] Actionable findings: 43 → 0

## Technical Quality Gates
- [x] Pre-commit linting (Ruff check/format) passed
- [x] Repository standards checked and verified
- [x] Zero deprecated / local absolute `file:///` URLs

## Review & Acceptance
- **Overall Verification Score**: 0%
- **Final Review Status**: **Needs Revision**
