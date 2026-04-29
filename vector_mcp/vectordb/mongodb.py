#!/usr/bin/python

from typing import Any

from agent_utilities import create_embedding_model
from llama_index.core import (
    VectorStoreIndex,
)

from vector_mcp.vectordb.base import Document, ItemID, QueryResults, VectorDB
from vector_mcp.vectordb.db_utils import (
    get_logger,
    optional_import_block,
    require_optional_import,
)

with optional_import_block():
    from pymongo import MongoClient

logger = get_logger(__name__)


@require_optional_import(["pymongo", "llama_index"], "mongodb")
class MongoDBAtlasVectorDB(VectorDB):
    """A vector database that uses MongoDB Atlas as the backend via LlamaIndex."""

    def __init__(
        self,
        *,
        connection_string: str | None = None,
        host: str | int | None = None,
        port: str | int | None = None,
        dbname: str | None = None,
        username: str | None = None,
        password: str | None = None,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        metadata: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize the vector database."""
        self.collection_name = collection_name
        self.embed_model = embed_model or create_embedding_model()
        self.metadata = metadata or {}

        if connection_string:
            self.connection_string = connection_string
        else:
            if username and password:
                self.connection_string = f"mongodb://{username}:{password}@{host}:{port or 27017}/{dbname or ''}"
            else:
                self.connection_string = (
                    f"mongodb://{host}:{port or 27017}/{dbname or ''}"
                )

        self.dbname = dbname or "default_db"
        self.mongo_client = MongoClient(self.connection_string)

        # For local MongoDB testing, we'll use a simpler approach
        # MongoDBAtlasVectorSearch is designed for MongoDB Atlas cloud, not local instances
        # For now, we'll implement basic vector operations using the raw client
        # This is a temporary solution for testing purposes
        self._use_simple_client = True
        self.storage_context = None
        self.vector_store = None
        self.active_collection = collection_name
        self.type = "mongodb"
        self._index = None

    def _get_collection(self, collection_name: str):
        """Get MongoDB collection for a given name."""
        return self.mongo_client[self.dbname][collection_name]

    def _get_index(self) -> "VectorStoreIndex":
        # For simple client approach, we'll implement basic operations without LlamaIndex index
        return None

    def create_collection(
        self, collection_name: str, overwrite: bool = False, _get_or_create: bool = True
    ) -> Any:
        self.collection_name = collection_name
        if overwrite:
            self._get_collection(collection_name).drop()
        self.active_collection = collection_name
        return self._get_collection(collection_name)

    def get_collection(self, collection_name: str | None = None) -> Any:
        return self._get_collection(collection_name or self.collection_name)

    def insert_documents(
        self,
        docs: list[Document],
        collection_name: str | None = None,
        _upsert: bool = False,
        **kwargs,
    ) -> None:
        collection = self._get_collection(collection_name or self.collection_name)
        for doc in docs:
            doc_id = doc.get("id")
            text = doc.get("content")
            metadata = doc.get("metadata", {})
            embedding = doc.get("embedding")

            if embedding is None:
                embedding = self.embed_model.get_text_embedding(text)

            document = {
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "embedding": embedding,
            }

            if _upsert:
                collection.update_one({"id": doc_id}, {"$set": document}, upsert=True)
            else:
                collection.insert_one(document)

    def semantic_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        distance_threshold: float = -1,
        **kwargs: Any,
    ) -> QueryResults:
        collection = self._get_collection(collection_name or self.collection_name)
        results = []

        import math

        for query in queries:
            query_embedding = self.embed_model.get_query_embedding(query)
            query_result = []

            # Get all documents and calculate cosine similarity
            documents = list(
                collection.find({}, {"embedding": 1, "text": 1, "metadata": 1, "id": 1})
            )

            for doc in documents:
                doc_embedding = doc.get("embedding", [])
                if not doc_embedding:
                    continue

                # Calculate cosine similarity
                dot_product = sum(
                    q * d for q, d in zip(query_embedding, doc_embedding, strict=True)
                )
                magnitude_q = math.sqrt(sum(q * q for q in query_embedding))
                magnitude_d = math.sqrt(sum(d * d for d in doc_embedding))

                if magnitude_q == 0 or magnitude_d == 0:
                    continue

                similarity = dot_product / (magnitude_q * magnitude_d)
                distance = 1.0 - similarity  # Convert similarity to distance

                if distance_threshold >= 0 and distance > distance_threshold:
                    continue

                query_result.append(
                    (
                        Document(
                            id=doc.get("id"),
                            content=doc.get("text"),
                            metadata=doc.get("metadata", {}),
                            embedding=doc.get("embedding"),
                        ),
                        similarity,
                    )
                )

            # Sort by similarity (descending) and limit results
            query_result.sort(key=lambda x: x[1], reverse=True)
            query_result = query_result[:n_results]

            results.append(query_result)

        return results

    def get_documents_by_ids(
        self,
        ids: list[ItemID] | None = None,
        collection_name: str | None = None,
        include=None,
        **kwargs,
    ) -> list[Document]:
        coll = self.get_collection(collection_name)
        if not ids:
            cursor = coll.find({})
        else:
            cursor = coll.find({"id": {"$in": ids}})

        docs = []
        for res in cursor:
            docs.append(
                Document(
                    id=res.get("id", str(res.get("_id"))),
                    content=res.get("text", ""),
                    metadata=res.get("metadata", {}),
                )
            )
        return docs

    def update_documents(
        self, docs: list[Document], collection_name: str | None = None, **kwargs
    ) -> None:
        self.insert_documents(docs, collection_name, upsert=True, **kwargs)

    def delete_documents(
        self, ids: list[ItemID], collection_name: str | None = None, **kwargs
    ) -> None:
        collection = self._get_collection(collection_name or self.collection_name)
        collection.delete_many({"id": {"$in": ids}})

    def delete_collection(self, collection_name: str) -> None:
        self._get_collection(collection_name).drop()
        if self.active_collection == collection_name:
            self.active_collection = ""

    def get_collections(self) -> Any:
        return self.mongo_client[self.dbname].list_collection_names()

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> QueryResults:
        coll = self.get_collection(collection_name)
        results = []
        for query in queries:
            try:
                pipeline = [
                    {
                        "$search": {
                            "index": "default",
                            "text": {"query": query, "path": "text"},
                        }
                    },
                    {"$limit": n_results},
                    {
                        "$project": {
                            "text": 1,
                            "metadata": 1,
                            "score": {"$meta": "searchScore"},
                        }
                    },
                ]

                cursor = coll.aggregate(pipeline)

                query_result = []
                for res in cursor:
                    doc = Document(
                        id=str(res.get("_id")),
                        content=res.get("text", ""),
                        metadata=res.get("metadata", {}),
                        embedding=None,
                    )
                    query_result.append((doc, res.get("score", 0.0)))
                results.append(query_result)
            except Exception as e:
                logger.error(f"MongoDB search failed: {e}")
                results.append([])

        return results
