import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Clear vector_mcp.vectordb modules from sys.modules cache to prevent caching issues
for k in list(sys.modules.keys()):
    if k.startswith("vector_mcp.vectordb"):
        sys.modules.pop(k, None)

# Save original modules to restore later
_orig_pgvector = sys.modules.get("llama_index.vector_stores.postgres")

# Define mock objects
mock_pgvector = MagicMock()
mock_pymongo = MagicMock()

# Place temporary mock for pgvector so the optional import succeeds
sys.modules["llama_index.vector_stores.postgres"] = mock_pgvector

# Import targets
from vector_mcp.vectordb.base import VectorDBFactory
from vector_mcp.vectordb.postgres import PostgreSQL
from vector_mcp.vectordb.mongodb import MongoDBAtlasVectorDB

# Restore original modules
if _orig_pgvector is not None:
    sys.modules["llama_index.vector_stores.postgres"] = _orig_pgvector
else:
    sys.modules.pop("llama_index.vector_stores.postgres", None)


@pytest.fixture(autouse=True)
def mock_mongodb_client():
    """Mock the MongoClient inside the mongodb module to avoid real connections."""
    with patch("vector_mcp.vectordb.mongodb.MongoClient", new=mock_pymongo.MongoClient) as mock:
        yield mock



def test_vectordb_factory():
    """Verify that VectorDBFactory correctly instantiates target databases."""
    # Test Chroma DB
    with patch("vector_mcp.vectordb.chromadb.ChromaVectorDB") as MockChroma:
        VectorDBFactory.create_vector_database("chroma", param="val")
        MockChroma.assert_called_once_with(param="val")

    # Test MongoDB
    with patch("vector_mcp.vectordb.mongodb.MongoDBAtlasVectorDB") as MockMongo:
        VectorDBFactory.create_vector_database("mongodb", param="val")
        MockMongo.assert_called_once_with(param="val")

    # Test Postgres
    with patch("vector_mcp.vectordb.postgres.PostgreSQL") as MockPostgres:
        VectorDBFactory.create_vector_database("postgres", param="val")
        MockPostgres.assert_called_once_with(param="val")

    # Test Qdrant
    with patch("vector_mcp.vectordb.qdrant.QdrantVectorDB") as MockQdrant:
        VectorDBFactory.create_vector_database("qdrant", param="val")
        MockQdrant.assert_called_once_with(param="val")

    # Test Couchbase
    with patch("vector_mcp.vectordb.couchbase.CouchbaseVectorDB") as MockCouchbase:
        VectorDBFactory.create_vector_database("couchbase", param="val")
        MockCouchbase.assert_called_once_with(param="val")

    # Test Unsupported
    with pytest.raises(ValueError, match="Unsupported vector database type"):
        VectorDBFactory.create_vector_database("invalid")


def test_mongodb_init():
    """Verify MongoDB Atlas client connection setup."""
    mock_pymongo.MongoClient.reset_mock()
    db = MongoDBAtlasVectorDB(
        host="localhost",
        port=27017,
        dbname="testdb",
        username="user",
        password="pwd",
        collection_name="col1",
    )
    assert db.collection_name == "col1"
    assert db.dbname == "testdb"
    assert db.type == "mongodb"
    mock_pymongo.MongoClient.assert_called_once_with(
        "mongodb://user:pwd@localhost:27017/testdb"
    )


def test_mongodb_operations():
    """Verify MongoDB Atlas CRUD operations using the simple client interface."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection

    mock_pymongo.MongoClient.return_value = mock_client

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding.return_value = [0.5, 0.5]
    mock_embed_model.get_query_embedding.return_value = [0.5, 0.5]

    db = MongoDBAtlasVectorDB(
        host="localhost",
        port=27017,
        dbname="testdb",
        collection_name="col1",
        embed_model=mock_embed_model,
    )

    # 1. create_collection
    db.create_collection("col2", overwrite=True)
    mock_collection.drop.assert_called_once()
    assert db.active_collection == "col2"

    # 2. insert_documents
    doc1 = {"id": "1", "content": "text1", "metadata": {"key": "val"}}
    db.insert_documents([doc1])
    mock_collection.insert_one.assert_called_once()

    # 3. delete_documents
    db.delete_documents(["1"])
    mock_collection.delete_many.assert_called_once_with({"id": {"$in": ["1"]}})

    # 4. get_collections
    mock_client.__getitem__.return_value.list_collection_names.return_value = [
        "c1",
        "c2",
    ]
    cols = db.get_collections()
    assert cols == ["c1", "c2"]

    # 5. get_documents_by_ids
    mock_collection.find.return_value = [{"id": "1", "text": "text1", "metadata": {}}]
    docs = db.get_documents_by_ids(ids=["1"])
    assert len(docs) == 1
    assert docs[0]["id"] == "1"
    assert docs[0]["content"] == "text1"

    # 6. semantic_search fallback
    mock_collection.find.return_value = [
        {"id": "1", "text": "text1", "embedding": [0.5, 0.5]}
    ]
    results = db.semantic_search(queries=["query1"])
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0][0]["id"] == "1"

    # 7. lexical_search aggregation
    mock_collection.aggregate.return_value = [
        {"_id": "1", "text": "text1", "score": 1.5}
    ]
    lex_res = db.lexical_search(queries=["query1"], n_results=5)
    assert len(lex_res) == 1
    assert lex_res[0][0][0]["content"] == "text1"
    assert lex_res[0][0][1] == 1.5


def test_postgres_init():
    """Verify PostgreSQL/PGVectorStore connection parameter setup."""
    mock_pgvector.PGVectorStore.from_params.reset_mock()
    db = PostgreSQL(
        connection_string="postgresql://user:pwd@localhost:5432/testdb",
        collection_name="col1",
    )
    assert db.collection_name == "col1"
    assert db.type == "postgres"
    mock_pgvector.PGVectorStore.from_params.assert_called_once()


def test_postgres_operations():
    """Verify PGVector database query formatting, inserts, and retrieves using SQLAlchemy connections."""
    mock_store = MagicMock()
    mock_pgvector.PGVectorStore.from_params.return_value = mock_store

    db = PostgreSQL(
        host="localhost",
        port=5432,
        dbname="testdb",
        collection_name="col1",
    )

    mock_index = MagicMock()

    # 1. create_collection
    with (
        patch.object(db, "_get_index", return_value=mock_index),
        patch.object(db, "get_collections", return_value=["col1"]),
    ):
        db.create_collection("col1")
        assert db.active_collection == "col1"

    # 2. insert_documents
    doc1 = {"id": "1", "content": "text1", "metadata": {"key": "val"}}
    with patch.object(db, "_get_index", return_value=mock_index):
        db.insert_documents([doc1])
        mock_index.insert.assert_called_once()

    # 3. delete_documents
    db.delete_documents(["1"])
    mock_store.delete_nodes.assert_called_once_with(["1"])

    # 4. get_documents_by_ids using SQLAlchemy Engine
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        (1, "text1", {"doc_id": "1", "ref_doc_id": "ref1"})
    ]
    mock_conn.execute.return_value = mock_result
    mock_store._engine.connect.return_value.__enter__.return_value = mock_conn

    docs = db.get_documents_by_ids(ids=["1"])
    assert len(docs) == 1
    assert docs[0]["id"] == "1"
    assert docs[0]["content"] == "text1"

    # 5. semantic_search using mocked LlamaIndex retriever nodes
    mock_node = MagicMock()
    mock_node.node.node_id = "1"
    mock_node.node.text = "text1"
    mock_node.node.metadata = {}
    mock_node.node.embedding = [0.1]
    mock_node.score = 0.2

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [mock_node]
    mock_index.as_retriever.return_value = mock_retriever

    with patch.object(db, "_get_index", return_value=mock_index):
        results = db.semantic_search(queries=["query1"])
        assert len(results) == 1
        assert results[0][0][0]["id"] == "1"
        assert results[0][0][1] == 0.8  # 1.0 - 0.2
