"""Mocked-client tests for the factory dispatch and the MongoDB/PostgreSQL providers.

The MongoDB and PostgreSQL backends were rewritten (raw pymongo/psycopg behind the
shared TLS-pinned transport layer, ``uri``/``tls_profile``-only construction — see
``vector_mcp/vectordb/mongodb.py`` / ``postgres.py``) as part of the security-hardening
pass; these tests exercise the current, real constructor/CRUD surface rather than the
superseded llama-index-backed implementation. Connection-security-specific behavior
(TLS enforcement, secret handling) is covered in ``test_secure_backends.py`` and
``test_database_transport_security.py`` — this file focuses on the CRUD/query paths.
Chroma/Couchbase are a deliberately-removed backend (see
``test_protocol_compliance.py::test_factory_rejects_removed_backend`` and
``test_database_transport_security.py``) and are not re-tested here.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vector_mcp.vectordb.base import VectorDBFactory


def _tls():
    return SimpleNamespace(
        verify_enabled=True,
        pymongo_kwargs=lambda: {"tls": True},
        psycopg_kwargs=lambda: {"sslmode": "verify-full"},
    )


def _embedder():
    model = MagicMock()
    model.get_query_embedding.return_value = [0.5, 0.5]
    model.get_text_embedding.return_value = [0.5, 0.5]
    # document_embeddings() prefers a batch API when present/callable; a bare
    # MagicMock auto-creates one, so disable it to force the per-item fallback.
    model.get_text_embedding_batch = None
    return model


def test_vectordb_factory():
    """Verify that VectorDBFactory correctly instantiates current-contract backends."""
    with patch("vector_mcp.vectordb.mongodb.MongoDBAtlasVectorDB") as MockMongo:
        VectorDBFactory.create_vector_database("mongodb", param="val")
        MockMongo.assert_called_once_with(param="val")

    with patch("vector_mcp.vectordb.postgres.PostgreSQL") as MockPostgres:
        VectorDBFactory.create_vector_database("postgres", param="val")
        MockPostgres.assert_called_once_with(param="val")

    with patch("vector_mcp.vectordb.qdrant.QdrantVectorDB") as MockQdrant:
        VectorDBFactory.create_vector_database("qdrant", param="val")
        MockQdrant.assert_called_once_with(param="val")

    with pytest.raises(ValueError, match="vector_database_type_unsupported"):
        VectorDBFactory.create_vector_database("invalid")


def test_mongodb_init(monkeypatch):
    """Verify MongoDB Atlas client connection setup with the current uri/tls_profile contract."""
    from vector_mcp.vectordb import mongodb

    constructor = MagicMock()
    monkeypatch.setattr(mongodb, "MongoClient", constructor)

    db = mongodb.MongoDBAtlasVectorDB(
        uri="mongodb://user:pwd@localhost:27017",
        dbname="testdb",
        tls_profile=_tls(),
        collection_name="col1",
        embed_model=_embedder(),
    )
    assert db.collection_name == "col1"
    assert db.dbname == "testdb"
    assert db.type == "mongodb"
    constructor.assert_called_once()
    assert constructor.call_args.args[0] == "mongodb://user:pwd@localhost:27017"
    assert constructor.call_args.kwargs["tls"] is True


def test_mongodb_operations(monkeypatch):
    """Verify MongoDB Atlas CRUD operations using the current raw-pymongo interface."""
    from vector_mcp.vectordb import mongodb

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
    monkeypatch.setattr(mongodb, "MongoClient", MagicMock(return_value=mock_client))

    db = mongodb.MongoDBAtlasVectorDB(
        uri="mongodb://user:pwd@localhost:27017",
        dbname="testdb",
        tls_profile=_tls(),
        collection_name="col1",
        embed_model=_embedder(),
    )

    # 1. insert_documents (upsert path -> bulk_write)
    doc1 = {"id": "1", "content": "text1", "metadata": {"key": "val"}}
    db.insert_documents([doc1], _upsert=True)
    mock_collection.bulk_write.assert_called_once()

    # 2. delete_documents
    db.delete_documents(["1"])
    mock_collection.delete_many.assert_called_once_with(
        {"document_id": {"$in": ["1"]}}
    )

    # 3. get_documents_by_ids
    mock_collection.find.return_value = [
        {"document_id": "1", "content": "text1", "metadata": {}}
    ]
    docs = db.get_documents_by_ids(ids=["1"])
    assert len(docs) == 1
    assert docs[0]["id"] == "1"
    assert docs[0]["content"] == "text1"

    # 4. semantic_search (native $vectorSearch aggregation)
    mock_collection.aggregate.return_value = [
        {"document_id": "1", "content": "text1", "metadata": {}, "score": 0.9}
    ]
    results = db.semantic_search(queries=["query1"])
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0][0]["id"] == "1"
    assert results[0][0][1] == pytest.approx(0.9)

    # 5. lexical_search ($text find)
    cursor = MagicMock()
    cursor.sort.return_value.limit.return_value = [
        {"document_id": "1", "content": "text1", "metadata": {}, "score": 1.5}
    ]
    mock_collection.find.return_value = cursor
    lex_res = db.lexical_search(queries=["query1"], n_results=5)
    assert len(lex_res) == 1
    assert lex_res[0][0][0]["content"] == "text1"
    assert lex_res[0][0][1] == 1.5


def test_postgres_init(monkeypatch):
    """Verify PostgreSQL connection-pool construction with the current structured-credential contract."""
    from vector_mcp.vectordb import postgres

    pool = MagicMock()
    monkeypatch.setattr(postgres, "ConnectionPool", MagicMock(return_value=pool))

    db = postgres.PostgreSQL(
        host="localhost",
        port=5432,
        dbname="testdb",
        username="user",
        password="pwd",
        tls_profile=_tls(),
        collection_name="col1",
        embed_model=_embedder(),
    )
    assert db.collection_name == "col1"
    assert db.type == "postgres"


def test_postgres_operations(monkeypatch):
    """Verify PostgreSQL query formatting, inserts, and retrieves against a mocked pool connection."""
    from vector_mcp.vectordb import postgres

    monkeypatch.setattr(postgres, "ConnectionPool", MagicMock(return_value=MagicMock()))

    db = postgres.PostgreSQL(
        host="localhost",
        port=5432,
        dbname="testdb",
        username="user",
        password="pwd",
        tls_profile=_tls(),
        collection_name="col1",
        embed_model=_embedder(),
    )

    cursor = MagicMock()
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    @contextmanager
    def connection_scope():
        yield connection

    db._connection = connection_scope
    db._exists = lambda _connection, _name: True

    # 1. insert_documents
    doc1 = {"id": "1", "content": "text1", "metadata": {"key": "val"}}
    db.insert_documents([doc1])
    cursor.executemany.assert_called_once()
    connection.commit.assert_called()

    # 2. delete_documents
    db.delete_documents(["1"])
    cursor.execute.assert_called()
    assert cursor.execute.call_args.args[1] == (["1"],)

    # 3. get_documents_by_ids
    cursor.fetchall.return_value = [("1", "text1", {"key": "val"})]
    docs = db.get_documents_by_ids(ids=["1"])
    assert len(docs) == 1
    assert docs[0]["id"] == "1"
    assert docs[0]["content"] == "text1"

    # 4. semantic_search
    cursor.fetchall.return_value = [("1", "text1", {}, 0.8)]
    results = db.semantic_search(queries=["query1"])
    assert len(results) == 1
    assert results[0][0][0]["id"] == "1"
    assert results[0][0][1] == pytest.approx(0.8)

    # 5. lexical_search
    cursor.fetchall.return_value = [("1", "text1", {}, 1.2)]
    lex_results = db.lexical_search(queries=["query1"])
    assert len(lex_results) == 1
    assert lex_results[0][0][0]["id"] == "1"
    assert lex_results[0][0][1] == pytest.approx(1.2)
