# AGENTS.md

## Scope

This repository provides the action-routed `vector-mcp` server, optional vector provider
adapters, an agent entry point, packaged skills/prompts, a vector ontology, and a read-only
collection-inventory connector preset.

- Language/Version: Python 3.11+
- Core Libraries: `agent-utilities`, `fastmcp`, `pydantic-ai`
- Key principles: Functional patterns, Pydantic for data validation, asynchronous tool execution.

## Current architecture

```text
MCP client
  -> vector_collection_management / vector_search
  -> validation, privacy, and backend policy
  -> verified GraphSession boundary
  -> configured provider
```

Epistemic-graph is the native default. Optional providers are isolated behind extras and imported
only when selected.

## Non-negotiable design rules

- Do not add legacy paths, fallback implementations, deprecated aliases, compatibility shims, or
  backward-compatibility branches. Migrate the current contract directly.
- Do not accept credentials or local database paths as MCP tool arguments.
- Do not persist endpoints, credentials, personal identity, raw content, hostnames, or local
  filesystem paths in source, skills, docs, traces, reports, fixtures, or generated evidence.
- Resolve runtime values through AgentConfig, environment variables, and supported secret
  references.
- Resolve TLS through the shared transport profile. Do not add certificate-verification bypasses.
- Keep document ingestion confined to the configured root and relative caller-selected paths.
- Keep optional dependencies lazy; importing `vector_mcp` must not start services or require
  every provider SDK.
- Never fabricate connector signatures, live test evidence, trace evidence, or safe dependency
  versions.

## Source layout

- `vector_mcp/mcp_server.py`: condensed MCP registration and request boundary
- `vector_mcp/vector_api.py`: verified session, privacy, secret, TLS, and provider boundary
- `vector_mcp/vectordb/`: optional provider adapters
- `vector_mcp/backend_policy.py`: canonical backend and exposure policy
- `vector_mcp/document_inputs.py`: bounded root-confined ingestion inputs
- `vector_mcp/doctor.py`: privacy-safe readiness output
- `vector_mcp/skills/`, `prompts/`, `ontology/`, `connectors/`: packaged extensions
- `tests/`: unit and explicitly configured integration tests
- `docs/`: operator documentation

### Supported backends

- **Epistemic-Graph** (**DEFAULT**): the native local AI-native engine — durable
  (redb-authoritative) with a native ANN/HNSW index, so semantic search is the engine's
  O(log N) vector search. Zero external infra (autostarts). Selected when `db_type` is
  unspecified (override with `DATABASE_TYPE`, alias `VECTOR_DB_TYPE`).
- **PostgreSQL/PGVector**, **Qdrant**, **MongoDB**: secure opt-in providers, TLS-verified and
  secret-reference-only — the current, certified contract (`VectorDBFactory.PREDEFINED_VECTOR_DB`
  and `backend_policy.ensure_backend_available` agree on exactly these four names, and both the
  factory and the MCP tool boundary reject any other `db_type`, including legacy spellings like
  `eg`/`epistemic-graph`).
- **ChromaDB / Couchbase — removed from the supported contract.** `vector_mcp/vectordb/chromadb.py`
  and `couchbase.py` remain in the tree (real, previously-shipped adapters) but are no longer wired
  into `VectorDBFactory`, `vectordb.__init__`'s exports, or the packaged extras —
  `db_type="chroma"|"couchbase"` raises `vector_database_type_unsupported` (see
  `tests/test_protocol_compliance.py`, `tests/test_database_transport_security.py`). They can still
  be imported directly (`from vector_mcp.vectordb.chromadb import ChromaVectorDB`) by a caller who
  installs `chromadb`/`couchbase` + `llama-index` themselves, but that path is unsupported/untested
  by the current contract.

### Implementation notes

- **Epistemic-Graph**: stores each chunk as an engine node (text in the indexed
  `description` property, plus metadata) with its embedding in the engine's native ANN
  index via `SyncEpistemicGraphClient` (transactional `.txn.add_node` + `.txn.add_embedding`,
  session-authority-cached connections). Semantic search is `.graph.semantic_search` (native
  ANN); lexical/keyword search is the engine's one-round-trip `.graph.discover` (keyword overlap
  + semantic, scalable — a client-side term scan is only a fallback for an engine too old for
  `discover`). A collection maps to an engine graph (`graph_name`); embeddings come from the
  shared `create_embedding_model`.
- **PostgreSQL / Qdrant / MongoDB**: rewritten on raw `psycopg`/`qdrant_client`/`pymongo` (no
  `llama-index` dependency) behind the shared TLS-pinned transport layer.

Deleted root-level debug scripts and obsolete test compose artifacts must remain deleted.
Maintained container definitions live under `docker/`.

## Development rules

- Use `agent-utilities` primitives for MCP construction, AgentConfig, secrets, transport
  security, and observability.
- Keep public tool schemas bounded and explicit.
- Use Pydantic fields/models where they define a public validation boundary.
- Log stable status and exception types, not values or response bodies.
- Require runtime credentials for integration tests; never supply checked-in password defaults.
- Preserve unrelated user work and do not commit caches, databases, traces, logs, build outputs,
  or environment files.
- Keep `pyproject.toml`, module versions, and `uv.lock` synchronized. Do not regenerate the
  lock as a side effect of unrelated work.

## Cheap validation

These checks do not start providers:

```bash
python scripts/security_sanitizer.py
python scripts/security_contract.py --contract .security/security-contract.json validate
python -m compileall -q vector_mcp
```

Also parse JSON/TOML/YAML, run `git diff --check`, and run the configured formatter/linter.
Provider tests, services, lock regeneration, and native compilation are separate serialized gates
and must not be run concurrently on a constrained workstation.

## Connector evidence

The source preset follows the installed action-routed tool schema. Exact tool-schema fingerprints,
connector manifests, and certification files are generated release evidence. Generate them only
after observing the installed MCP schema and resolving an authorized runtime signing key. If
source ontology or tool schemas change, old signatures are invalid and must not be copied forward.

## ⛔ Keep the Repository Root Pristine — No Scratch / Temp / Debug Files

**The repository ROOT must contain only canonical project files** (packaging,
config, docs, lockfiles). The only hidden directories allowed at root are
`.git/`, `.github/`, and `.specify/` (plus a local, git-ignored `.venv/`).

**NEVER write any of the following — anywhere in the repo, and ESPECIALLY at the root:**
- One-off / debug / migration scripts: `fix_*.py`, `migrate_*.py`, `refactor_*.py`,
  `replace_*.py`, `update_*.py`, `debug_*.py`, or `test_*.py` **at the root**
  (real tests live in `tests/` only).
- Databases / data dumps: `*.db`, `*.db-wal`, `*.sqlite*`, `*.corrupted`.
- Logs / command output: `*.log`, scratch `*.txt`, `*.orig`, `*.rej`, `*.bak`.
- Build artifacts: `*.tsbuildinfo`, compiled binaries, coverage files.
- AI agent scratch directories: `.agent/`, `.agents/`, `.agent_data/`, `.tmp/`,
  `.hypothesis/`, or any per-tool cache committed to git.
- Any file that is NOT production source, a test in `tests/`, documentation, or
  a recognized config/lockfile.

**Why:** these files expose private filesystem paths, credentials, and internal infrastructure
details when pushed to GitHub publicly; scratch at the root also bloats the tree and erodes a
pristine codebase.

**Where to put scratch work instead:**
- Use `~/workspace/scratch/` for temporary scripts and experiments
- Use `~/workspace/reports/` for command output and reports
- Keep test scripts in the `tests/` directory following proper pytest conventions

Before finishing a task, run `git status` and confirm no stray root files were added.

<!-- BEGIN concept-coordination (generated) -->
## Concept-ID Coordination (multi-session)

Working in parallel with other sessions/worktrees? **Reserve a concept id before you write its `CONCEPT:` marker** so two sessions never collide:

```bash
agent-utilities --json concept reserve --ns EG-KG.compute.backend   # or a package prefix, e.g. KEY
```

Full protocol (ledger, merge=union, reconcile, MCP/REST): <https://knuckles-team.github.io/agent-utilities/concept_coordination/>
<!-- END concept-coordination (generated) -->

## Version & lockfile drift edict (keep the version mirrors AND the lock in sync)

The two most common release-breakers in this fleet are **version drift** (the version in
`pyproject.toml`/`.bumpversion.cfg` advancing while `README.md`, `docker/Dockerfile`, and the
module `__version__`s lag) and a **stale `uv.lock`** (shipping known-vulnerable transitive deps).
A version mismatch makes the next `bump-my-version` throw `VersionNotFoundException`; a stale lock
is what Dependabot flags. Rules:

1. **Never hand-edit a version string.** Change the version ONLY via
   `bump-my-version bump {patch|minor|major}` (a.k.a. `bump2version`), which rewrites every file
   registered in `.bumpversion.cfg` in one atomic, tagged commit. If you edited the version in
   `pyproject.toml` by hand, you created drift — revert and use the bumper.
2. **Every version-bearing file must be registered in `.bumpversion.cfg`** — at minimum
   `pyproject.toml` AND `README.md`, plus `docker/Dockerfile` and any module `__version__`. Never
   add a file that embeds the version without a `[bumpversion:file:...]` entry for it.
3. **Re-lock on every dependency change.** After editing `pyproject.toml` deps/extras, run
   `uv lock` and commit `uv.lock` in the SAME change. The `uv-lock` pre-commit hook runs with
   `--locked` and fails on drift — never bypass it. The committed `uv.lock` is the
   Dependabot/security surface.
4. **Patch CVEs with a version floor at the source, then re-lock.** `uv` resolves one version
   graph-wide, so a lower-bound in the extra that pulls a dependency raises it for the whole lock.
