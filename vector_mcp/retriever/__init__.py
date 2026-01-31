#!/usr/bin/python
# coding: utf-8

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chromadb_retriever import ChromaDBRetriever
    from .llamaindex_retriever import LlamaIndexRetriever
    from .mongodb_retriever import MongoDBRetriever
    from .pgvector_retriever import PGVectorRetriever
    from .couchbase_retriever import CouchbaseRetriever
    from .qdrant_retriever import QdrantRetriever
    from .retriever import RAGRetriever

__all__ = [
    "ChromaDBRetriever",
    "LlamaIndexRetriever",
    "MongoDBRetriever",
    "RAGRetriever",
    "PGVectorRetriever",
    "CouchbaseRetriever",
    "QdrantRetriever",
]


def __getattr__(name: str):
    if name == "ChromaDBRetriever":
        from .chromadb_retriever import ChromaDBRetriever

        return ChromaDBRetriever
    elif name == "LlamaIndexRetriever":
        from .llamaindex_retriever import LlamaIndexRetriever

        return LlamaIndexRetriever
    elif name == "MongoDBRetriever":
        from .mongodb_retriever import MongoDBRetriever

        return MongoDBRetriever
    elif name == "PGVectorRetriever":
        from .pgvector_retriever import PGVectorRetriever

        return PGVectorRetriever
    elif name == "CouchbaseRetriever":
        from .couchbase_retriever import CouchbaseRetriever

        return CouchbaseRetriever
    elif name == "QdrantRetriever":
        from .qdrant_retriever import QdrantRetriever

        return QdrantRetriever
    elif name == "RAGRetriever":
        from .retriever import RAGRetriever

        return RAGRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
