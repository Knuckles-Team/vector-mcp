import pytest
from vector_mcp.vectordb.postgres import PostgreSQL
from vector_mcp.vectordb.mongodb import MongoDBAtlasVectorDB
from vector_mcp.vectordb.couchbase import CouchbaseVectorDB
from vector_mcp.vectordb.qdrant import QdrantVectorDB


@pytest.fixture
def sample_docs():
    return [
        {"id": "1", "content": "Test doc 1", "metadata": {"key": "val1"}},
        {"id": "2", "content": "Test doc 2", "metadata": {"key": "val2"}},
    ]


@pytest.fixture
def pgvector_db():
    try:
        db = PostgreSQL(
            connection_string="postgresql://postgres:password@localhost:5432/vectordb",
            collection_name="test_collection",
        )
        return db
    except Exception as e:
        pytest.skip(f"PGVector not available: {e}")


def test_pgvector_integration(pgvector_db, sample_docs):

    try:
        pgvector_db.create_collection("test_col_int", overwrite=True)
        pgvector_db.insert_documents(sample_docs)
        results = pgvector_db.semantic_search(["Test"])
        assert len(results) > 0
    except Exception as e:
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            pytest.skip(f"PGVector connection failed during op: {e}")
        else:
            import psycopg

            if isinstance(e, psycopg.OperationalError):
                pytest.skip("PGVector connection refused")
            raise e


@pytest.fixture
def mongo_db():
    try:
        db = MongoDBAtlasVectorDB(
            host="localhost",
            port=27017,
            dbname="vectordb",
            username="mongo",
            password="password",
            collection_name="test_col_mongo",
        )
        return db
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")


def test_mongodb_integration(mongo_db, sample_docs):
    try:
        mongo_db.mongo_client.admin.command("ping")
    except Exception as e:
        pytest.skip(f"MongoDB connection failed: {e}")

    mongo_db.create_collection("test_col_mongo", overwrite=True)
    mongo_db.insert_documents(sample_docs)
    count = mongo_db.get_collection("test_col_mongo").count_documents({})
    assert count == 2


@pytest.fixture
def qdrant_db():
    try:
        db = QdrantVectorDB(
            location="http://localhost:6333",
            collection_name="test_col_qdrant",
        )
        return db
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")


def test_qdrant_integration(qdrant_db, sample_docs):
    try:
        qdrant_db.client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant connection failed: {e}")

    qdrant_db.create_collection("test_col_qdrant", overwrite=True)
    qdrant_db.insert_documents(sample_docs)

    import time

    time.sleep(1)

    results = qdrant_db.semantic_search(["Test"])
    assert len(results) > 0


@pytest.fixture
def couchbase_db():
    try:
        db = CouchbaseVectorDB(
            connection_string="couchbase://localhost",
            username="Administrator",
            password="password",
            dbname="vector_db",
            collection_name="test_col_cb",
        )
        return db
    except Exception as e:
        pytest.skip(f"Couchbase connection init failed: {e}")


def test_couchbase_integration(couchbase_db, sample_docs):
    try:
        couchbase_db.cluster.ping()
    except Exception as e:
        pytest.skip(f"Couchbase ping failed: {e}")

    couchbase_db.insert_documents(sample_docs, collection_name="test_col_cb")
