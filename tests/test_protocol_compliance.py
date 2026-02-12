import pytest
from typing import Protocol, runtime_checkable
from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb.base import VectorDB

from vector_mcp.retriever.chromadb_retriever import ChromaDBRetriever
from vector_mcp.retriever.pgvector_retriever import PGVectorRetriever
from vector_mcp.retriever.couchbase_retriever import CouchbaseRetriever
from vector_mcp.retriever.mongodb_retriever import MongoDBRetriever
from vector_mcp.retriever.qdrant_retriever import QdrantRetriever

from vector_mcp.vectordb.chromadb import ChromaVectorDB
from vector_mcp.vectordb.postgres import PostgreSQL
from vector_mcp.vectordb.couchbase import CouchbaseVectorDB
from vector_mcp.vectordb.mongodb import MongoDBAtlasVectorDB
from vector_mcp.vectordb.qdrant import QdrantVectorDB


def test_rag_retriever_is_protocol():
    assert issubclass(RAGRetriever, Protocol)


def test_vectordb_is_protocol():
    assert issubclass(VectorDB, Protocol)


@pytest.mark.parametrize(
    "retriever_cls",
    [
        ChromaDBRetriever,
        PGVectorRetriever,
        CouchbaseRetriever,
        MongoDBRetriever,
        QdrantRetriever,
    ],
)
def test_retriever_implements_protocol(retriever_cls):

    assert issubclass(retriever_cls, RAGRetriever)


@pytest.mark.parametrize(
    "vectordb_cls",
    [
        ChromaVectorDB,
        PostgreSQL,
        CouchbaseVectorDB,
        MongoDBAtlasVectorDB,
        QdrantVectorDB,
    ],
)
def test_vectordb_implements_protocol(vectordb_cls):
    assert issubclass(vectordb_cls, VectorDB)


def test_no_super_init_usage():
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
                        if "retriever.py" in file or "base.py" in file:
                            pass
                        elif "test" in file:
                            pass
                        else:
                            if file in [
                                "chromadb_retriever.py",
                                "postgres_retriever.py",
                                "chromadb.py",
                                "postgres.py",
                            ]:
                                pytest.fail(f"Found super().__init__ in {file}")
