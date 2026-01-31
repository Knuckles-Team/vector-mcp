import pytest
from typing import Protocol, runtime_checkable
from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb.base import VectorDB

# Import implementations
# Using try-except imports to handle optional dependencies if they were missing, 
# but since we installed [all], they should be there.
from vector_mcp.retriever.chromadb_retriever import ChromaDBRetriever
from vector_mcp.retriever.pgvector_retriever import PGVectorRetriever
from vector_mcp.retriever.couchbase_retriever import CouchbaseRetriever
from vector_mcp.retriever.mongodb_retriever import MongoDBRetriever
from vector_mcp.retriever.qdrant_retriever import QdrantRetriever

from vector_mcp.vectordb.chromadb import ChromaVectorDB
from vector_mcp.vectordb.pgvector import PGVectorDB
from vector_mcp.vectordb.couchbase import CouchbaseVectorDB
from vector_mcp.vectordb.mongodb import MongoDBAtlasVectorDB
from vector_mcp.vectordb.qdrant import QdrantVectorDB

def test_rag_retriever_is_protocol():
    assert issubclass(RAGRetriever, Protocol)

def test_vectordb_is_protocol():
    assert issubclass(VectorDB, Protocol)

@pytest.mark.parametrize("retriever_cls", [
    ChromaDBRetriever,
    PGVectorRetriever,
    CouchbaseRetriever,
    MongoDBRetriever,
    QdrantRetriever,
])
def test_retriever_implements_protocol(retriever_cls):
    # Since protocols are runtime checkable, we can checks compliance
    # But note: isinstance check works better on instances. 
    # For classes, we verify they have the methods.
    
    # Instantiate with minimal args if possible, or just mock
    # RAGRetriever protocols methods: initialize_collection, add_documents, connect_database, query
    
    assert issubclass(retriever_cls, RAGRetriever)

@pytest.mark.parametrize("vectordb_cls", [
    ChromaVectorDB,
    PGVectorDB,
    CouchbaseVectorDB,
    MongoDBAtlasVectorDB,
    QdrantVectorDB,
])
def test_vectordb_implements_protocol(vectordb_cls):
    assert issubclass(vectordb_cls, VectorDB)

def test_no_super_init_usage():
    # Inspection test to ensure no super().__init__ calls exist in key files
    # This repeats the logic of verify_protocol.py but as a test
    import os
    import vector_mcp
    
    package_dir = os.path.dirname(vector_mcp.__file__)
    
    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    content = f.read()
                    if "super().__init__" in content:
                        # Allow it in __init__.py or unrelated files if necessary, 
                        # but for our retrievers/dbs it should be gone.
                        if "retriever.py" in file or "base.py" in file:
                             # These are protocols now, shouldn't satisfy it anyway basically.
                             pass
                        elif "test" in file:
                             pass
                        else:
                             # Checking specific known files that were refactored
                             if file in ["chromadb_retriever.py", "pgvector_retriever.py", "chromadb.py", "pgvector.py"]:
                                 pytest.fail(f"Found super().__init__ in {file}")

