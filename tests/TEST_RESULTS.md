# Vector provider validation matrix

The repository gate is intentionally network-free. Passing it demonstrates the current
client contract and configuration wiring, not availability of any deployment.

| Capability | Native engine | PostgreSQL | MongoDB Atlas | Qdrant |
| --- | --- | --- | --- | --- |
| Current factory route | covered | covered | covered | covered |
| Runtime-only credentials | not required | covered | covered | covered |
| Shared verified TLS | engine resolver | covered | covered | covered + DNS pin |
| Indexed semantic search | delegated ANN | HNSW | Atlas vector index | Qdrant vector index |
| Indexed lexical search | engine discovery | GIN `tsvector` | text index | full-text payload index |
| Bounded batch ingestion | batched existence + embeddings, ACID commit | batched embeddings + `executemany` | batched embeddings + bulk write | batched existence + embeddings + upsert |
| Cached dimension probe | engine-owned schema | process-local O(1) cache | process-local O(1) cache | process-local O(1) cache |
| Readiness polling | engine resolver | immediate catalog check | bounded exponential backoff | synchronous index creation |
| Hybrid top-k fusion | O(k log limit) bounded heap | O(k log limit) bounded heap | O(k log limit) bounded heap | O(k log limit) bounded heap |
| CRUD contract | mocked | mocked | mocked | mocked |
| No live network calls | pass | pass | pass | pass |

Removed providers and legacy aliases are tested as rejected inputs. Live certification
must be recorded separately with only revision, provider version, status, counts, and
opaque trace identifiers—never endpoints, credentials, identities, paths, queries, or
retrieved content.
