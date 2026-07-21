"""Secure PostgreSQL/pgvector backend with HNSW and GIN indexes."""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Generator
from contextlib import contextmanager
from functools import cached_property
from typing import Any

from agent_utilities import create_embedding_model
from agent_utilities.core.transport_security import ResolvedTLSProfile

from vector_mcp.vectordb.base import (
    Document,
    ItemID,
    QueryResults,
    VectorDB,
    document_embeddings,
)
from vector_mcp.vectordb.db_utils import optional_import_block, require_optional_import

with optional_import_block():
    from psycopg import sql
    from psycopg.types.json import Jsonb
    from psycopg_pool import ConnectionPool

_CATALOG = "vector_mcp_collections"


def _table_name(collection_name: str) -> str:
    digest = hashlib.sha256(collection_name.encode("utf-8")).hexdigest()[:24]
    return f"vm_{digest}"


def _vector(values: Any) -> str:
    vector = [float(value) for value in values]
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("embedding_invalid")
    return "[" + ",".join(format(value, ".17g") for value in vector) + "]"


@require_optional_import(["psycopg", "psycopg_pool"], "postgres")
class PostgreSQL(VectorDB):
    """Direct pgvector provider; every query is parameterized and index-backed."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        dbname: str,
        username: str,
        password: str,
        tls_profile: ResolvedTLSProfile,
        timeout: int = 30,
        max_pool_size: int = 20,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        if not tls_profile.verify_enabled:
            raise ValueError("postgres_tls_profile_invalid")
        self.collection_name = collection_name
        self.embed_model = embed_model or create_embedding_model()
        self.metadata = metadata or {}
        self.type = "postgres"
        self.active_collection = collection_name
        connect_kwargs = {
            "host": host,
            "port": int(port),
            "dbname": dbname,
            "user": username,
            "password": password,
            "connect_timeout": int(timeout),
            **tls_profile.psycopg_kwargs(),
        }
        self._pool = ConnectionPool(
            kwargs=connect_kwargs,
            min_size=0,
            max_size=int(max_pool_size),
            timeout=float(timeout),
            open=False,
        )
        self._open_lock = threading.Lock()
        self._opened = False
        self._catalog_ready = False

    def _open(self) -> None:
        if self._opened:
            return
        with self._open_lock:
            if not self._opened:
                self._pool.open(wait=True)
                self._opened = True

    @contextmanager
    def _connection(self) -> Generator[Any, None, None]:
        self._open()
        with self._pool.connection() as connection:
            if not self._catalog_ready:
                with connection.cursor() as cursor:
                    cursor.execute(
                        sql.SQL(
                            "CREATE TABLE IF NOT EXISTS {} ("
                            "collection_name TEXT PRIMARY KEY, "
                            "table_name TEXT UNIQUE NOT NULL, "
                            "dimension INTEGER NOT NULL CHECK (dimension > 0))"
                        ).format(sql.Identifier(_CATALOG))
                    )
                connection.commit()
                self._catalog_ready = True
            yield connection

    @cached_property
    def _embedding_dimension(self) -> int:
        return len(self.embed_model.get_query_embedding("vector dimension probe"))

    def _dimension(self) -> int:
        return self._embedding_dimension

    @staticmethod
    def _document(row: Any) -> Document:
        return Document(
            id=str(row[0]),
            content=str(row[1]),
            metadata=dict(row[2] or {}),
        )

    def _exists(self, connection: Any, collection_name: str) -> bool:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT 1 FROM {} WHERE collection_name = %s").format(
                    sql.Identifier(_CATALOG)
                ),
                (collection_name,),
            )
            return cursor.fetchone() is not None

    def create_collection(
        self, collection_name: str, overwrite: bool = False, _get_or_create: bool = True
    ) -> str:
        table_name = _table_name(collection_name)
        dimension = self._dimension()
        with self._connection() as connection, connection.cursor() as cursor:
            exists = self._exists(connection, collection_name)
            if exists and overwrite:
                cursor.execute(
                    sql.SQL("DROP TABLE {}").format(sql.Identifier(table_name))
                )
                cursor.execute(
                    sql.SQL("DELETE FROM {} WHERE collection_name = %s").format(
                        sql.Identifier(_CATALOG)
                    ),
                    (collection_name,),
                )
                exists = False
            if exists and not _get_or_create:
                raise ValueError("collection_exists")
            if exists:
                cursor.execute(
                    sql.SQL(
                        "SELECT dimension FROM {} WHERE collection_name = %s"
                    ).format(sql.Identifier(_CATALOG)),
                    (collection_name,),
                )
                row = cursor.fetchone()
                if row is None or int(row[0]) != dimension:
                    raise ValueError("collection_vector_schema_mismatch")
            if not exists:
                cursor.execute(
                    sql.SQL(
                        "CREATE TABLE {} ("
                        "id TEXT PRIMARY KEY, content TEXT NOT NULL, "
                        "metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, "
                        "embedding VECTOR({}) NOT NULL)"
                    ).format(sql.Identifier(table_name), sql.SQL(str(dimension)))
                )
                cursor.execute(
                    sql.SQL(
                        "CREATE INDEX {} ON {} USING hnsw (embedding vector_cosine_ops)"
                    ).format(
                        sql.Identifier(f"{table_name}_hnsw"), sql.Identifier(table_name)
                    )
                )
                cursor.execute(
                    sql.SQL(
                        "CREATE INDEX {} ON {} USING gin "
                        "(to_tsvector('simple', content))"
                    ).format(
                        sql.Identifier(f"{table_name}_text"), sql.Identifier(table_name)
                    )
                )
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} (collection_name, table_name, dimension) "
                        "VALUES (%s, %s, %s)"
                    ).format(sql.Identifier(_CATALOG)),
                    (collection_name, table_name, dimension),
                )
            connection.commit()
        self.collection_name = collection_name
        self.active_collection = collection_name
        return collection_name

    def get_collection(self, collection_name: str | None = None) -> str:
        name = collection_name or self.collection_name
        with self._connection() as connection:
            if not self._exists(connection, name):
                raise ValueError("collection_not_found")
        return name

    def get_collections(self) -> list[str]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "SELECT collection_name FROM {} ORDER BY collection_name"
                ).format(sql.Identifier(_CATALOG))
            )
            return [str(row[0]) for row in cursor.fetchall()]

    def delete_collection(self, collection_name: str) -> None:
        table_name = _table_name(collection_name)
        with self._connection() as connection, connection.cursor() as cursor:
            if not self._exists(connection, collection_name):
                raise ValueError("collection_not_found")
            cursor.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table_name)))
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE collection_name = %s").format(
                    sql.Identifier(_CATALOG)
                ),
                (collection_name,),
            )
            connection.commit()
        if self.active_collection == collection_name:
            self.active_collection = ""

    def insert_documents(
        self,
        docs: list[Document],
        collection_name: str | None = None,
        _upsert: bool = False,
        **_kwargs: Any,
    ) -> None:
        name = collection_name or self.collection_name
        table_name = _table_name(name)
        rows = []
        vectors = document_embeddings(docs, self.embed_model)
        for document, embedding in zip(docs, vectors, strict=True):
            content = str(document["content"])
            rows.append(
                (
                    str(document["id"]),
                    content,
                    Jsonb(dict(document.get("metadata") or {})),
                    _vector(embedding),
                )
            )
        statement = sql.SQL(
            "INSERT INTO {} (id, content, metadata, embedding) "
            "VALUES (%s, %s, %s, %s::vector)"
        ).format(sql.Identifier(table_name))
        if _upsert:
            statement += sql.SQL(
                " ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, "
                "metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding"
            )
        with self._connection() as connection, connection.cursor() as cursor:
            if not self._exists(connection, name):
                raise ValueError("collection_not_found")
            if rows:
                cursor.executemany(statement, rows)
            connection.commit()

    def update_documents(
        self, docs: list[Document], collection_name: str | None = None, **kwargs: Any
    ) -> None:
        self.insert_documents(docs, collection_name, _upsert=True, **kwargs)

    def delete_documents(
        self, ids: list[ItemID], collection_name: str | None = None, **_kwargs: Any
    ) -> None:
        table_name = _table_name(collection_name or self.collection_name)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(
                    sql.Identifier(table_name)
                ),
                ([str(value) for value in ids],),
            )
            connection.commit()

    def get_documents_by_ids(
        self,
        ids: list[ItemID] | None = None,
        collection_name: str | None = None,
        include: list[str] | None = None,
        **_kwargs: Any,
    ) -> list[Document]:
        del include
        if not ids:
            raise ValueError("document_ids_required")
        table_name = _table_name(collection_name or self.collection_name)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "SELECT id, content, metadata FROM {} WHERE id = ANY(%s)"
                ).format(sql.Identifier(table_name)),
                ([str(value) for value in ids],),
            )
            return [self._document(row) for row in cursor.fetchall()]

    def semantic_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        distance_threshold: float = -1,
        **_kwargs: Any,
    ) -> QueryResults:
        table_name = _table_name(collection_name or self.collection_name)
        results: QueryResults = []
        with self._connection() as connection, connection.cursor() as cursor:
            for query in queries:
                vector = _vector(self.embed_model.get_query_embedding(query))
                where = (
                    sql.SQL("WHERE embedding <=> q.value <= %s")
                    if distance_threshold >= 0
                    else sql.SQL("")
                )
                statement = sql.SQL(
                    "WITH q AS (SELECT %s::vector AS value) "
                    "SELECT id, content, metadata, 1 - (embedding <=> q.value) AS score "
                    "FROM {} CROSS JOIN q {} ORDER BY embedding <=> q.value LIMIT %s"
                ).format(sql.Identifier(table_name), where)
                parameters: tuple[Any, ...] = (
                    (vector, float(distance_threshold), int(n_results))
                    if distance_threshold >= 0
                    else (vector, int(n_results))
                )
                cursor.execute(statement, parameters)
                results.append(
                    [(self._document(row), float(row[3])) for row in cursor.fetchall()]
                )
        return results

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        **_kwargs: Any,
    ) -> QueryResults:
        table_name = _table_name(collection_name or self.collection_name)
        statement = sql.SQL(
            "WITH q AS (SELECT plainto_tsquery('simple', %s) AS value) "
            "SELECT id, content, metadata, "
            "ts_rank_cd(to_tsvector('simple', content), q.value) AS score "
            "FROM {} CROSS JOIN q "
            "WHERE to_tsvector('simple', content) @@ q.value "
            "ORDER BY score DESC LIMIT %s"
        ).format(sql.Identifier(table_name))
        results: QueryResults = []
        with self._connection() as connection, connection.cursor() as cursor:
            for query in queries:
                cursor.execute(statement, (query, int(n_results)))
                results.append(
                    [(self._document(row), float(row[3])) for row in cursor.fetchall()]
                )
        return results

    def close(self) -> None:
        if self._opened:
            self._pool.close()
