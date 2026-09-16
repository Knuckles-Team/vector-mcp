# Backing Platform — Vector Databases

`vector-mcp` is a **client** of a vector database. ChromaDB runs entirely on the
local filesystem and needs no external service; the other supported backends —
**PostgreSQL/PGVector**, **Qdrant**, **MongoDB**, and **Couchbase** — connect to a
deployed database. This page provides Docker recipes for deploying one locally to
serve as the target of the connector's `*_URL` / credential settings. For production
topologies, follow each database's upstream documentation.

!!! note "Backing-system recipe"
    Each connector in the ecosystem follows the same convention — a
    `docs/platform.md` recipe for the system it integrates with, accompanied by a
    sample Compose stack. Backends offered only as a managed service have no local
    recipe.

## Single-node deployment (Compose)

The repository's [`docker/compose.test.yml`](https://github.com/Knuckles-Team/vector-mcp/blob/main/docker/compose.test.yml)
runs every container-backed store on one network. Deploy the one you need:

=== "PostgreSQL / PGVector"

    ```yaml
    # docker/postgres.compose.yml
    services:
      postgres:
        image: docker.io/paradedb/paradedb@sha256:ac8ad72f57da6fa35738d346b9970811a4ab9ce0ba7f183bea27e0ad7b037222  # paradedb, originally tagged `latest-pg16`
        container_name: vector-postgres
        restart: unless-stopped
        environment:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: password
          POSTGRES_DB: vectordb
        ports:
          - "5432:5432"
        volumes:
          - pgdata:/var/lib/postgresql/data
        healthcheck:
          test: ["CMD-SHELL", "pg_isready -U postgres"]
          interval: 5s
          timeout: 5s
          retries: 10

    volumes:
      pgdata:
    ```

=== "Qdrant"

    ```yaml
    # docker/qdrant.compose.yml
    services:
      qdrant:
        image: docker.io/qdrant/qdrant@sha256:12364fe851b9f17356fc88189fc06d1b521262e04659ec7345975b00c9246a10  # qdrant/qdrant, originally tagged `latest`
        container_name: vector-qdrant
        restart: unless-stopped
        ports:
          - "6333:6333"
          - "6334:6334"
        volumes:
          - qdrant_data:/qdrant/storage

    volumes:
      qdrant_data:
    ```

=== "MongoDB"

    ```yaml
    # docker/mongodb.compose.yml
    services:
      mongodb:
        image: docker.io/mongo@sha256:2609aaf7a1abbff404101af896e05f243d22be742471ed857b50b9ce0270fdbd  # mongo, originally tagged `latest`
        container_name: vector-mongodb
        restart: unless-stopped
        ports:
          - "27017:27017"
        volumes:
          - mongo_data:/data/db

    volumes:
      mongo_data:
    ```

=== "Couchbase"

    ```yaml
    # docker/couchbase.compose.yml
    services:
      couchbase:
        image: docker.io/couchbase@sha256:c4254e37f5a5db3d13eb990d89862765db44b4f96b6283190c2e6c8a20d760d7  # couchbase, originally tagged `latest`
        container_name: vector-couchbase
        restart: unless-stopped
        environment:
          COUCHBASE_ADMINISTRATOR_USERNAME: Administrator
          COUCHBASE_ADMINISTRATOR_PASSWORD: password
          COUCHBASE_BUCKET: vector_db
          COUCHBASE_BUCKET_RAMSIZE: 256
        ports:
          - "8091:8091"
          - "8093:8093"
          - "11210:11210"
        volumes:
          - couchbase_data:/opt/couchbase/var

    volumes:
      couchbase_data:
    ```

```bash
docker compose -f docker/postgres.compose.yml up -d
docker exec vector-postgres pg_isready -U postgres        # wait until ready
```

## Connect vector-mcp

Point the connector at the deployed database with the matching backend type and
connection settings. For PostgreSQL/PGVector:

```bash
export DATABASE_TYPE=postgres
export DB_HOST=localhost
export DB_PORT=5432
export DBNAME=vectordb
export USERNAME=postgres
export PASSWORD=password
export EMBEDDING_MODEL_ID=text-embedding-nomic-embed-text-v2-moe

vector-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

ChromaDB requires no service — set `DATABASE_TYPE=chromadb` and a `DATABASE_PATH`
on the local filesystem.

## Combined deployment

A combined stack places the database and the MCP server on one Docker network, so the
server reaches the store by container name:

```yaml
# docker/stack.compose.yml
services:
  postgres:
    image: docker.io/paradedb/paradedb@sha256:ac8ad72f57da6fa35738d346b9970811a4ab9ce0ba7f183bea27e0ad7b037222  # paradedb, originally tagged `latest-pg16`
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: vectordb
    volumes: ["pgdata:/var/lib/postgresql/data"]

  vector-mcp:
    image: knucklessg1/vector-mcp@sha256:<digest>  # resolve via `docker manifest inspect knucklessg1/vector-mcp:mcp` or your release process
    depends_on: [postgres]
    environment:
      - DATABASE_TYPE=postgres
      - DB_HOST=postgres
      - DB_PORT=5432
      - DBNAME=vectordb
      - USERNAME=postgres
      - PASSWORD=password
      - TRANSPORT=streamable-http
      - HOST=0.0.0.0
      - PORT=8000
    ports: ["8000:8000"]

volumes:
  pgdata:
```

```bash
docker compose -f docker/stack.compose.yml up -d
```
