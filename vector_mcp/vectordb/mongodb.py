#!/usr/bin/python
# coding: utf-8
from typing import Any, Optional, Union

from vector_mcp.vectordb.base import Document, ItemID, QueryResults, VectorDB
from vector_mcp.vectordb.utils import (
    get_logger,
    optional_import_block,
    require_optional_import,
)

from vector_mcp.utils import get_embedding_model

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document as LIDocument,
)

with optional_import_block():
    from pymongo import MongoClient
    from llama_index.vector_stores.mongodb import MongoDBVectorStore

logger = get_logger(__name__)


@require_optional_import(["pymongo", "llama_index"], "retrievechat-mongodb")
class MongoDBAtlasVectorDB(VectorDB):
    """A vector database that uses MongoDB Atlas as the backend via LlamaIndex."""

    def __init__(
        self,
        *,
        connection_string: Optional[str] = None,
        host: Optional[Union[str, int]] = None,
        port: Optional[Union[str, int]] = None,
        dbname: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Initialize the vector database."""
        self.collection_name = collection_name
        self.embed_model = embed_model or get_embedding_model()
        self.metadata = metadata or {}

        # Construct connection string
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

        self.vector_store = MongoDBVectorStore(
            mongo_client=self.mongo_client,
            db_name=self.dbname,
            collection_name=self.collection_name,
            **kwargs,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.active_collection = collection_name
        self.type = "mongodb"
        self._index = None

    def _get_index(self) -> "VectorStoreIndex":
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
            )
        return self._index

    def create_collection(
        self, collection_name: str, overwrite: bool = False, get_or_create: bool = True
    ) -> Any:
        self.collection_name = collection_name
        if overwrite:
            # Drop collection
            db = self.mongo_client[self.dbname]
            db.drop_collection(collection_name)

        self.vector_store = MongoDBVectorStore(
            mongo_client=self.mongo_client,
            db_name=self.dbname,
            collection_name=self.collection_name,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.active_collection = collection_name
        self._index = None
        return self.vector_store

    def get_collection(self, collection_name: str = None) -> Any:
        db = self.mongo_client[self.dbname]
        return db[collection_name or self.collection_name]

    def insert_documents(
        self,
        docs: list[Document],
        collection_name: str = None,
        upsert: bool = False,
        **kwargs,
    ) -> None:
        if collection_name:
            self.create_collection(collection_name)

        li_docs = []
        for doc in docs:
            metadata = doc.get("metadata", {}) or {}
            li_docs.append(
                LIDocument(text=doc["content"], doc_id=doc["id"], metadata=metadata)
            )

        index = self._get_index()
        for li_doc in li_docs:
            index.insert(li_doc)

    def semantic_search(
        self,
        queries: list[str],
        collection_name: str = None,
        n_results: int = 10,
        distance_threshold: float = -1,
        **kwargs: Any,
    ) -> QueryResults:
        if collection_name:
            self.create_collection(collection_name)

        index = self._get_index()
        results = []
        retriever = index.as_retriever(similarity_top_k=n_results)

        for query in queries:
            nodes = retriever.retrieve(query)
            query_result = []
            for node_match in nodes:
                if distance_threshold >= 0 and node_match.score < distance_threshold:
                    continue

                doc = Document(
                    id=node_match.node.node_id,
                    content=node_match.node.text,
                    metadata=node_match.node.metadata,
                    embedding=node_match.node.embedding,
                )
                query_result.append((doc, 1.0 - (node_match.score or 0.0)))
            results.append(query_result)
        return results

    def get_documents_by_ids(
        self,
        ids: list[ItemID] = None,
        collection_name: str = None,
        include=None,
        **kwargs,
    ) -> list[Document]:
        # Simple mongo find
        coll = self.get_collection(collection_name)
        if not ids:
            cursor = coll.find({})
        else:
            cursor = coll.find(
                {"id": {"$in": ids}}
            )  # Assuming 'id' is store field, LlamaIndex uses 'id_' or 'doc_id' typically but stores metadata

        # NOTE: LlamaIndex schema might differ, but assuming basic storage
        docs = []
        for res in cursor:
            # This depends on LlamaIndex internal schema in MongoDB
            docs.append(
                Document(
                    id=res.get("id", str(res.get("_id"))),
                    content=res.get("text", ""),
                    metadata=res.get("metadata", {}),
                )
            )
        return docs

    def update_documents(
        self, docs: list[Document], collection_name: str = None
    ) -> None:
        self.insert_documents(docs, collection_name, upsert=True)

    def delete_documents(
        self, ids: list[ItemID], collection_name: str = None, **kwargs
    ) -> None:
        if collection_name:
            self.create_collection(collection_name)
        self.vector_store.delete_nodes(ids)

    def delete_collection(self, collection_name: str) -> None:
        db = self.mongo_client[self.dbname]
        db.drop_collection(collection_name)
        if self.active_collection == collection_name:
            self.active_collection = None

    def get_collections(self) -> Any:
        db = self.mongo_client[self.dbname]
        return db.list_collection_names()

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> QueryResults:
        coll = self.get_collection(collection_name)
        results = []
        for query in queries:
            try:
                # MongoDB Atlas Search (Atlas only, not standard Mongo usually, but user mentioned mongot/atlas search)
                # Requires an index definition on the collection.
                # Pipeline:
                # [{
                #   "$search": {
                #     "index": "default", # or custom
                #     "text": {
                #       "query": query,
                #       "path": "text" # LlamaIndex usually stores content in 'text' field (or 'doc_content'?)
                #     }
                #   }
                # }, {
                #   "$limit": n_results
                # }, {
                #   "$project": { ... }
                # }]

                # Check LlamaIndex MongoDB schema:
                # content is usually "text" key.

                pipeline = [
                    {
                        "$search": {
                            "index": "default",  # User must ensure this index exists
                            "text": {"query": query, "path": "text"},  # Targeted field
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
                        id=str(res.get("_id")),  # or res.get("id_") or similar
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
