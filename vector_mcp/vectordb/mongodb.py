"""Secure MongoDB Atlas vector backend with indexed semantic and lexical search."""

from __future__ import annotations

import time
from functools import cached_property
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from agent_utilities import create_embedding_model
from agent_utilities.core.transport_security import ResolvedTLSProfile

from vector_mcp.vectordb.base import (
    Document,
    ItemID,
    QueryResults,
    VectorDB,
    document_embeddings,
)
from vector_mcp.vectordb.db_utils import optional_import_block, require_optional_import

with optional_import_block():
    from pymongo import ASCENDING, TEXT, MongoClient
    from pymongo.operations import ReplaceOne, SearchIndexModel

_VECTOR_INDEX = "vector_mcp_embedding"
_TEXT_INDEX = "vector_mcp_content"


def _validate_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if parsed.scheme not in {"mongodb", "mongodb+srv"} or not parsed.hostname:
        raise ValueError("mongodb_uri_invalid")
    if parsed.fragment:
        raise ValueError("mongodb_uri_invalid")
    forbidden = {
        "tls",
        "ssl",
        "tlsallowinvalidcertificates",
        "tlsallowinvalidhostnames",
        "tlsinsecure",
    }
    if any(key.casefold() in forbidden for key, _value in parse_qsl(parsed.query)):
        raise ValueError("mongodb_uri_tls_override_forbidden")
    if not parsed.username or not parsed.password:
        raise ValueError("mongodb_uri_auth_required")
    return uri


@require_optional_import(["pymongo"], "mongodb")
class MongoDBAtlasVectorDB(VectorDB):
    """MongoDB Atlas provider; unsupported search-index deployments fail closed."""

    def __init__(
        self,
        *,
        uri: str,
        dbname: str,
        tls_profile: ResolvedTLSProfile,
        timeout_ms: int = 30_000,
        max_pool_size: int = 20,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        if not tls_profile.verify_enabled:
            raise ValueError("mongodb_tls_profile_invalid")
        self.collection_name = collection_name
        self.embed_model = embed_model or create_embedding_model()
        self.metadata = metadata or {}
        self.dbname = dbname
        self.type = "mongodb"
        self.active_collection = collection_name
        self._timeout_ms = int(timeout_ms)
        self.mongo_client = MongoClient(
            _validate_uri(uri),
            serverSelectionTimeoutMS=int(timeout_ms),
            connectTimeoutMS=int(timeout_ms),
            socketTimeoutMS=int(timeout_ms),
            maxPoolSize=int(max_pool_size),
            **tls_profile.pymongo_kwargs(),
        )

    def _collection(self, collection_name: str | None = None) -> Any:
        return self.mongo_client[self.dbname][collection_name or self.collection_name]

    @cached_property
    def _embedding_dimension(self) -> int:
        return len(self.embed_model.get_query_embedding("vector dimension probe"))

    def _dimension(self) -> int:
        return self._embedding_dimension

    @staticmethod
    def _vector_index_valid(index: dict[str, Any], dimension: int) -> bool:
        definition = index.get("latestDefinition") or index.get("definition") or {}
        fields = definition.get("fields") if isinstance(definition, dict) else None
        if not isinstance(fields, list):
            return False
        return any(
            isinstance(field, dict)
            and field.get("type") == "vector"
            and field.get("path") == "embedding"
            and field.get("numDimensions") == dimension
            and field.get("similarity") == "cosine"
            for field in fields
        )

    def _wait_for_vector_index(self, collection: Any, dimension: int) -> None:
        deadline = time.monotonic() + self._timeout_ms / 1_000
        poll_interval = 0.1
        while time.monotonic() < deadline:
            index = next(
                (
                    item
                    for item in collection.list_search_indexes(_VECTOR_INDEX)
                    if item.get("name") == _VECTOR_INDEX
                ),
                None,
            )
            if index is not None:
                if not self._vector_index_valid(index, dimension):
                    raise ValueError("collection_vector_schema_mismatch")
                status = str(index.get("status", "")).upper()
                if index.get("queryable") is True or status in {"READY", "STEADY"}:
                    return
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(poll_interval, remaining))
            poll_interval = min(poll_interval * 2, 1.0)
        raise ValueError("collection_vector_index_not_ready")

    def create_collection(
        self, collection_name: str, overwrite: bool = False, _get_or_create: bool = True
    ) -> Any:
        database = self.mongo_client[self.dbname]
        exists = collection_name in database.list_collection_names()
        if exists and overwrite:
            database.drop_collection(collection_name)
            exists = False
        if exists and not _get_or_create:
            raise ValueError("collection_exists")
        if not exists:
            database.create_collection(collection_name)
        collection = database[collection_name]
        collection.create_index(
            [("document_id", ASCENDING)], name="vector_mcp_document_id", unique=True
        )
        collection.create_index([("content", TEXT)], name=_TEXT_INDEX)

        dimension = self._dimension()
        existing = {item.get("name") for item in collection.list_search_indexes()}
        if _VECTOR_INDEX not in existing:
            definition = {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": dimension,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "document_id"},
                ]
            }
            collection.create_search_index(
                SearchIndexModel(
                    definition=definition, name=_VECTOR_INDEX, type="vectorSearch"
                )
            )
        self._wait_for_vector_index(collection, dimension)
        self.collection_name = collection_name
        self.active_collection = collection_name
        return collection

    def get_collection(self, collection_name: str | None = None) -> Any:
        return self._collection(collection_name)

    def get_collections(self) -> list[str]:
        return self.mongo_client[self.dbname].list_collection_names()

    def delete_collection(self, collection_name: str) -> None:
        self.mongo_client[self.dbname].drop_collection(collection_name)
        if self.active_collection == collection_name:
            self.active_collection = ""

    @staticmethod
    def _doc(value: dict[str, Any]) -> Document:
        return Document(
            id=str(value.get("document_id", "")),
            content=str(value.get("content", "")),
            metadata=dict(value.get("metadata") or {}),
        )

    def insert_documents(
        self,
        docs: list[Document],
        collection_name: str | None = None,
        _upsert: bool = False,
        **_kwargs: Any,
    ) -> None:
        collection = self._collection(collection_name)
        vectors = document_embeddings(docs, self.embed_model)
        values: list[dict[str, Any]] = []
        for document, embedding in zip(docs, vectors, strict=True):
            content = str(document["content"])
            value = {
                "document_id": str(document["id"]),
                "content": content,
                "metadata": dict(document.get("metadata") or {}),
                "embedding": embedding,
            }
            values.append(value)
        if not values:
            return
        if _upsert:
            collection.bulk_write(
                [
                    ReplaceOne(
                        {"document_id": value["document_id"]}, value, upsert=True
                    )
                    for value in values
                ],
                ordered=True,
            )
        else:
            collection.insert_many(values, ordered=True)

    def update_documents(
        self, docs: list[Document], collection_name: str | None = None, **kwargs: Any
    ) -> None:
        self.insert_documents(docs, collection_name, _upsert=True, **kwargs)

    def delete_documents(
        self, ids: list[ItemID], collection_name: str | None = None, **_kwargs: Any
    ) -> None:
        self._collection(collection_name).delete_many(
            {"document_id": {"$in": [str(value) for value in ids]}}
        )

    def get_documents_by_ids(
        self,
        ids: list[ItemID] | None = None,
        collection_name: str | None = None,
        include: list[str] | None = None,
        **_kwargs: Any,
    ) -> list[Document]:
        del include
        if not ids:
            raise ValueError("document_ids_required")
        cursor = self._collection(collection_name).find(
            {"document_id": {"$in": [str(value) for value in ids]}},
            {"document_id": 1, "content": 1, "metadata": 1},
        )
        return [self._doc(value) for value in cursor]

    def semantic_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        distance_threshold: float = -1,
        **_kwargs: Any,
    ) -> QueryResults:
        collection = self._collection(collection_name)
        results: QueryResults = []
        for query in queries:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": _VECTOR_INDEX,
                        "path": "embedding",
                        "queryVector": [
                            float(item)
                            for item in self.embed_model.get_query_embedding(query)
                        ],
                        "numCandidates": max(int(n_results) * 10, 100),
                        "limit": int(n_results),
                    }
                },
                {
                    "$project": {
                        "document_id": 1,
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            values: list[tuple[Document, float]] = []
            for item in collection.aggregate(pipeline):
                similarity = float(item.get("score", 0.0))
                if distance_threshold >= 0 and 1.0 - similarity > distance_threshold:
                    continue
                values.append((self._doc(item), similarity))
            results.append(values)
        return results

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        **_kwargs: Any,
    ) -> QueryResults:
        collection = self._collection(collection_name)
        results: QueryResults = []
        for query in queries:
            cursor = (
                collection.find(
                    {"$text": {"$search": query}},
                    {
                        "document_id": 1,
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "textScore"},
                    },
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(int(n_results))
            )
            results.append(
                [(self._doc(item), float(item.get("score", 0.0))) for item in cursor]
            )
        return results

    def close(self) -> None:
        self.mongo_client.close()
