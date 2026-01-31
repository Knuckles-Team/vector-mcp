#!/usr/bin/python
# coding: utf-8
from .base import Document, VectorDB
from .utils import get_logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pgvector import PGVectorDB
    from .qdrant import QdrantVectorDB
    from .couchbase import CouchbaseVectorDB
    from .mongodb import MongoDBAtlasVectorDB
    from .chromadb import ChromaVectorDB

__all__ = [
    "get_logger",
    "Document",
    "VectorDB",
    "PGVectorDB",
    "QdrantVectorDB",
    "CouchbaseVectorDB",
    "MongoDBAtlasVectorDB",
    "ChromaVectorDB",
]


def __getattr__(name: str):
    if name == "PGVectorDB":
        from .pgvector import PGVectorDB

        return PGVectorDB
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
