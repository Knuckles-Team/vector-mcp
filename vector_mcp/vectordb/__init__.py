#!/usr/bin/python

from typing import TYPE_CHECKING

from agent_utilities import get_logger

from .base import Document, VectorDB

if TYPE_CHECKING:
    from .epistemic_graph import EpistemicGraphVectorDB
    from .mongodb import MongoDBAtlasVectorDB
    from .postgres import PostgreSQL
    from .qdrant import QdrantVectorDB

__all__ = [
    "get_logger",
    "Document",
    "VectorDB",
    "EpistemicGraphVectorDB",
    "PostgreSQL",
    "QdrantVectorDB",
    "MongoDBAtlasVectorDB",
]


def __getattr__(name: str):
    if name == "EpistemicGraphVectorDB":
        from .epistemic_graph import EpistemicGraphVectorDB

        return EpistemicGraphVectorDB
    elif name == "PostgreSQL":
        from .postgres import PostgreSQL

        return PostgreSQL
    elif name == "QdrantVectorDB":
        from .qdrant import QdrantVectorDB

        return QdrantVectorDB
    elif name == "MongoDBAtlasVectorDB":
        from .mongodb import MongoDBAtlasVectorDB

        return MongoDBAtlasVectorDB
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
