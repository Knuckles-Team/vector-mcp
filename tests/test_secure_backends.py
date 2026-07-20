from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _tls():
    return SimpleNamespace(
        verify_enabled=True,
        httpx_kwargs=lambda: {"verify": "shared-context", "trust_env": False},
        pymongo_kwargs=lambda: {"tls": True, "tlsCAFile": "runtime-material"},
        psycopg_kwargs=lambda: {
            "sslmode": "verify-full",
            "sslrootcert": "runtime-material",
        },
    )


def _embedder():
    model = MagicMock()
    model.get_query_embedding.return_value = [0.1, 0.2, 0.3]
    model.get_text_embedding.return_value = [0.1, 0.2, 0.3]
    return model


def _batch_embedder():
    model = _embedder()
    model.get_text_embedding_batch.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    return model


def test_qdrant_constructor_forces_https_and_shared_tls(monkeypatch) -> None:
    from vector_mcp.vectordb import qdrant

    client = MagicMock()
    constructor = MagicMock(return_value=client)
    transport = object()
    embedder = _embedder()
    monkeypatch.setattr(qdrant, "QdrantClient", constructor)
    transport_factory = MagicMock(return_value=transport)
    monkeypatch.setattr(qdrant, "pinned_egress_transport", transport_factory)

    database = qdrant.QdrantVectorDB(
        host="vector.example.invalid",
        port=6333,
        api_key="runtime-only",
        allowed_private_hosts=["vector.example.invalid"],
        tls_profile=_tls(),
        embed_model=embedder,
    )

    kwargs = constructor.call_args.kwargs
    assert kwargs["url"] == "https://vector.example.invalid:6333"
    assert kwargs["prefer_grpc"] is False
    assert kwargs["verify"] == "shared-context"
    assert kwargs["trust_env"] is False
    assert kwargs["transport"] is transport
    assert kwargs["check_compatibility"] is False
    transport_factory.assert_called_once_with(
        verify="shared-context",
        allowed_private_hosts=["vector.example.invalid"],
        allow_loopback=False,
    )
    assert database._dimension() == database._dimension() == 3
    embedder.get_query_embedding.assert_called_once_with("vector dimension probe")
    assert database.type == "qdrant"


def test_qdrant_rejects_proxy_that_would_bypass_dns_pin(monkeypatch) -> None:
    from vector_mcp.vectordb import qdrant

    profile = SimpleNamespace(
        verify_enabled=True,
        httpx_kwargs=lambda: {
            "verify": "shared-context",
            "trust_env": True,
            "proxy": "https://proxy.example.invalid",
        },
    )
    monkeypatch.setattr(qdrant, "QdrantClient", MagicMock())

    with pytest.raises(ValueError, match="qdrant_proxy_unsupported"):
        qdrant.QdrantVectorDB(
            host="vector.example.invalid",
            port=6333,
            api_key="runtime-only",
            tls_profile=profile,
            embed_model=_embedder(),
        )


def test_qdrant_lexical_query_is_server_filtered(monkeypatch) -> None:
    from vector_mcp.vectordb import qdrant

    client = MagicMock()
    client.scroll.return_value = ([], None)
    monkeypatch.setattr(qdrant, "QdrantClient", MagicMock(return_value=client))
    monkeypatch.setattr(qdrant, "pinned_egress_transport", MagicMock())
    database = qdrant.QdrantVectorDB(
        host="vector.example.invalid",
        port=6333,
        api_key="runtime-only",
        tls_profile=_tls(),
        embed_model=_embedder(),
    )

    assert database.lexical_search(["indexed terms"], "docs", 4) == [[]]
    assert client.scroll.call_args.kwargs["limit"] == 4
    assert client.scroll.call_args.kwargs["scroll_filter"] is not None


def test_qdrant_ingestion_batches_existence_and_upsert(monkeypatch) -> None:
    from vector_mcp.vectordb import qdrant

    client = MagicMock()
    client.retrieve.return_value = []
    monkeypatch.setattr(qdrant, "QdrantClient", MagicMock(return_value=client))
    monkeypatch.setattr(qdrant, "pinned_egress_transport", MagicMock())
    database = qdrant.QdrantVectorDB(
        host="vector.example.invalid",
        port=6333,
        api_key="runtime-only",
        tls_profile=_tls(),
        embed_model=_batch_embedder(),
    )

    database.insert_documents(
        [
            {"id": "one", "content": "first"},
            {"id": "two", "content": "second"},
        ],
        "docs",
    )

    client.retrieve.assert_called_once()
    client.upsert.assert_called_once()
    assert len(client.upsert.call_args.kwargs["points"]) == 2


def test_mongodb_constructor_rejects_tls_downgrade(monkeypatch) -> None:
    from vector_mcp.vectordb import mongodb

    monkeypatch.setattr(mongodb, "MongoClient", MagicMock())
    with pytest.raises(ValueError, match="tls_override_forbidden"):
        mongodb.MongoDBAtlasVectorDB(
            uri="mongodb://database.example.invalid/?tls=false",
            dbname="vectors",
            tls_profile=_tls(),
            embed_model=_embedder(),
        )


def test_mongodb_constructor_applies_shared_tls(monkeypatch) -> None:
    from vector_mcp.vectordb import mongodb

    constructor = MagicMock()
    monkeypatch.setattr(mongodb, "MongoClient", constructor)
    mongodb.MongoDBAtlasVectorDB(
        uri="mongodb+srv://runtime:secret@database.example.invalid/vectors",
        dbname="vectors",
        tls_profile=_tls(),
        embed_model=_embedder(),
    )
    kwargs = constructor.call_args.kwargs
    assert kwargs["tls"] is True
    assert kwargs["tlsCAFile"] == "runtime-material"
    assert "username" not in kwargs
    assert "password" not in kwargs


def test_mongodb_ingestion_uses_one_bulk_write(monkeypatch) -> None:
    from vector_mcp.vectordb import mongodb

    collection = MagicMock()
    database = object.__new__(mongodb.MongoDBAtlasVectorDB)
    database.collection_name = "docs"
    database.embed_model = _batch_embedder()
    database._collection = MagicMock(return_value=collection)

    database.insert_documents(
        [
            {"id": "one", "content": "first"},
            {"id": "two", "content": "second"},
        ],
        _upsert=True,
    )

    collection.bulk_write.assert_called_once()
    assert len(collection.bulk_write.call_args.args[0]) == 2


def test_postgres_pool_receives_structured_credentials_and_verify_full(
    monkeypatch,
) -> None:
    from vector_mcp.vectordb import postgres

    pool = MagicMock()
    constructor = MagicMock(return_value=pool)
    monkeypatch.setattr(postgres, "ConnectionPool", constructor, raising=False)
    init = getattr(
        postgres.PostgreSQL.__init__, "__wrapped__", postgres.PostgreSQL.__init__
    )
    database = object.__new__(postgres.PostgreSQL)
    init(
        database,
        host="database.example.invalid",
        port=5432,
        dbname="vectors",
        username="runtime-user",
        password="runtime-password",
        tls_profile=_tls(),
        embed_model=_embedder(),
    )

    kwargs = constructor.call_args.kwargs["kwargs"]
    assert kwargs["host"] == "database.example.invalid"
    assert kwargs["sslmode"] == "verify-full"
    assert kwargs["user"] == "runtime-user"
    assert kwargs["password"] == "runtime-password"
    assert not any("postgresql://" in str(value) for value in kwargs.values())


def test_postgres_existing_collection_rejects_embedding_dimension_drift() -> None:
    from vector_mcp.vectordb import postgres

    database = object.__new__(postgres.PostgreSQL)
    database.collection_name = "docs"
    database.active_collection = "docs"
    database.embed_model = _embedder()
    database._exists = lambda _connection, _name: True

    cursor = MagicMock()
    cursor.fetchone.return_value = (999,)
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    @contextmanager
    def connection_scope():
        yield connection

    database._connection = connection_scope

    with pytest.raises(ValueError, match="collection_vector_schema_mismatch"):
        database.create_collection("docs")

    assert "SELECT dimension" in str(cursor.execute.call_args.args[0])


def test_native_backend_delegates_without_transport_arguments(monkeypatch) -> None:
    from vector_mcp.vectordb import epistemic_graph

    control = MagicMock()
    control.tenants.list.return_value = []
    connect = MagicMock(return_value=control)
    session = SimpleNamespace(
        engine_verified_context=lambda: {
            "principal": "opaque-actor",
            "tenant": "opaque-tenant",
            "audience": "agent-services",
            "agent_id": "opaque-actor",
            "roles": [],
            "scopes": ["kg:admin"],
            "delegation": [],
            "policy_version": "current",
        }
    )
    monkeypatch.setattr(epistemic_graph, "resolve_session", lambda: session)
    monkeypatch.setattr(
        epistemic_graph, "SyncEpistemicGraphClient", SimpleNamespace(connect=connect)
    )
    database = epistemic_graph.EpistemicGraphVectorDB(embed_model=_embedder())

    database.create_collection("knowledge")

    assert all(
        set(call.kwargs) == {"graph_name", "verified_context"}
        for call in connect.call_args_list
    )
    control.tenants.create.assert_called_once_with("knowledge", "Agent")


def test_native_ingestion_batches_existence_and_embeddings() -> None:
    from vector_mcp.vectordb import epistemic_graph

    client = MagicMock()
    client.nodes.has_batch.return_value = {"one": False, "two": False}
    client.txn.begin.return_value = "transaction"
    client.txn.commit.return_value = True
    embedder = _batch_embedder()
    database = epistemic_graph.EpistemicGraphVectorDB(embed_model=embedder)
    database._client_for = MagicMock(return_value=client)

    database.insert_documents(
        [
            {"id": "one", "content": "first"},
            {"id": "two", "content": "second"},
        ],
        "docs",
    )

    client.nodes.has_batch.assert_called_once_with(["one", "two"])
    embedder.get_text_embedding_batch.assert_called_once_with(["first", "second"])
    assert client.txn.add_embedding.call_count == 2
    client.txn.commit.assert_called_once_with("transaction")
