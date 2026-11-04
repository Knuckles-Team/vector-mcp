#!/usr/bin/python

import logging
from typing import TYPE_CHECKING

from .base import Document, VectorDB


def get_logger(name: str) -> logging.Logger:
    """Standard library logger, re-exported for backward compatibility."""
    return logging.getLogger(name)


if TYPE_CHECKING:
    from .medulla import MedullaVectorDB
    from .mongodb import MongoDBAtlasVectorDB
    from .postgres import PostgreSQL
    from .qdrant import QdrantVectorDB

__all__ = [
    "get_logger",
    "Document",
    "VectorDB",
    "MedullaVectorDB",
    "PostgreSQL",
    "QdrantVectorDB",
    "MongoDBAtlasVectorDB",
]


def __getattr__(name: str):
    if name == "MedullaVectorDB":
        from .medulla import MedullaVectorDB

        return MedullaVectorDB
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
