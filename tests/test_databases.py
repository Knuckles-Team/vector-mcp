import pytest
import os
from unittest.mock import MagicMock, patch
from vector_mcp.vectordb.pgvector import PGVectorDB
from vector_mcp.vectordb.mongodb import MongoDBAtlasVectorDB
from vector_mcp.vectordb.couchbase import CouchbaseVectorDB
from vector_mcp.vectordb.qdrant import QdrantVectorDB
from vector_mcp.vectordb.base import Document

# Integration tests that try to connect to real services.
# If connection fails, we skip.

@pytest.fixture
def sample_docs():
    return [
        {"id": "1", "content": "Test doc 1", "metadata": {"key": "val1"}},
        {"id": "2", "content": "Test doc 2", "metadata": {"key": "val2"}},
    ]

# --- PGVECTOR ---
@pytest.fixture
def pgvector_db():
    try:
        # Default Postgres params from compose.yml
        db = PGVectorDB(
            connection_string="postgresql://postgres:password@localhost:5432/vectordb",
            collection_name="test_collection"
        )
        # Try to connect (usually lazy, so we might need to trigger something)
        # PGVectorDB uses PGVectorStore which might connect on init or operation.
        # Verify connection by listing something or creating collection
        return db
    except Exception as e:
        pytest.skip(f"PGVector not available: {e}")

def test_pgvector_integration(pgvector_db, sample_docs):
    # This might fail if DB is not up, but fixture handles skip? 
    # Actually if fixture raises Skip, test is skipped.
    # But PGVectorDB init might not raise if connection is lazy.
    
    # We'll try an operation
    try:
        pgvector_db.create_collection("test_col_int", overwrite=True)
        pgvector_db.insert_documents(sample_docs)
        results = pgvector_db.semantic_search(["Test"])
        assert len(results) > 0
    except Exception as e:
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            pytest.skip(f"PGVector connection failed during op: {e}")
        else:
             # If it's a code error, fail.
             # But identifying connection error vs code error is hard without specific Exceptions.
             # We will assume environment is flaky and mark xfail or skip if specific error.
             # For now, let's just let it fail so user knows to fix env, 
             # UNLESS it is clearly a connection error.
             import psycopg
             if isinstance(e, psycopg.OperationalError):
                  pytest.skip("PGVector connection refused")
             raise e

# --- MONGODB ---
@pytest.fixture
def mongo_db():
    try:
        db = MongoDBAtlasVectorDB(
            host="localhost",
            port=27017,
            dbname="vectordb",
            username="mongo", # compose: MONGO_INITDB_ROOT_USERNAME=mongo
            password="password",
            collection_name="test_col_mongo"
        )
        return db
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")

def test_mongodb_integration(mongo_db, sample_docs):
    try:
        mongo_db.mongo_client.admin.command('ping')
    except Exception as e:
        pytest.skip(f"MongoDB connection failed: {e}")

    mongo_db.create_collection("test_col_mongo", overwrite=True)
    mongo_db.insert_documents(sample_docs)
    # Search requires index setup in Atlas usually, but local mongo?
    # MongoDBVectorStore in LlamaIndex might need specific setup.
    # We check if insert worked.
    count = mongo_db.get_collection("test_col_mongo").count_documents({})
    assert count == 2

# --- QDRANT ---
@pytest.fixture
def qdrant_db():
    try:
        db = QdrantVectorDB(
            location="http://localhost:6333", # Accessing HTTP port
            collection_name="test_col_qdrant"
        )
        return db
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")

def test_qdrant_integration(qdrant_db, sample_docs):
    try:
        qdrant_db.client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant connection failed: {e}")

    # Create & Insert
    qdrant_db.create_collection("test_col_qdrant", overwrite=True)
    qdrant_db.insert_documents(sample_docs)
    
    # Verify
    # Qdrant might take a moment to index?
    import time
    time.sleep(1)
    
    results = qdrant_db.semantic_search(["Test"])
    assert len(results) > 0

# --- COUCHBASE ---
@pytest.fixture
def couchbase_db():
    try:
        db = CouchbaseVectorDB(
            connection_string="couchbase://localhost",
            username="Administrator",
            password="password",
            dbname="vector_db",
            collection_name="test_col_cb"
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
    # Verify
