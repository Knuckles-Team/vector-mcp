#!/usr/bin/python

from typing import TYPE_CHECKING

from agent_utilities import get_logger

from .base import Document, VectorDB

if TYPE_CHECKING:
    from .chromadb import ChromaVectorDB
    from .couchbase import CouchbaseVectorDB
    from .mongodb import MongoDBAtlasVectorDB
    from .postgres import PostgreSQL
    from .qdrant import QdrantVectorDB

__all__ = [
    "get_logger",
    "Document",
    "VectorDB",
    "PostgreSQL",
    "QdrantVectorDB",
    "CouchbaseVectorDB",
    "MongoDBAtlasVectorDB",
    "ChromaVectorDB",
]


def __getattr__(name: str):
    if name == "PostgreSQL":
        from .postgres import PostgreSQL

        return PostgreSQL
    elif name == "QdrantVectorDB":
        from .qdrant import QdrantVectorDB

        return QdrantVectorDB
    elif name == "CouchbaseVectorDB":
        from .couchbase import CouchbaseVectorDB

        return CouchbaseVectorDB
    elif name == "MongoDBAtlasVectorDB":
        from .mongodb import MongoDBAtlasVectorDB

        return MongoDBAtlasVectorDB
    elif name == "ChromaVectorDB":
        from .chromadb import ChromaVectorDB

        return ChromaVectorDB
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
