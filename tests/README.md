# Vector provider test contract

The default suite is offline and resource-bounded. It validates:

- native epistemic-graph delegation without transport arguments;
- PostgreSQL structured credentials and `verify-full` TLS;
- MongoDB Atlas secret-held URI and mandatory shared TLS;
- Qdrant HTTPS construction, DNS-pinned egress, and indexed full-text filtering;
- batched embedding, existence-check, and provider-write boundaries;
- PII sanitization and removal of source/path metadata before persistence;
- action routing, document-root confinement, and bounded hybrid RRF;
- optional-provider failure isolation.

Run it with:

```bash
pytest -q
```

Provider SDK clients are mocked. The suite does not start containers, connect to
databases, compile the native engine, or use deployment credentials. Live provider
qualification belongs to the target deployment and must use AgentConfig, runtime secret
references, verified TLS profiles, least-privilege accounts, and sanitized evidence.
