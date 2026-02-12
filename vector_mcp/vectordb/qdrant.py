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
    import qdrant_client
    from qdrant_client import models
    from llama_index.vector_stores.qdrant import QdrantVectorStore

logger = get_logger(__name__)


@require_optional_import(["qdrant_client", "llama_index"], "retrievechat-qdrant")
class QdrantVectorDB(VectorDB):
    """A vector database that uses Qdrant as the backend via LlamaIndex."""

    def __init__(
        self,
        *,
        location: Optional[str] = None,
        url: Optional[str] = None,
        host: Optional[Union[str, int]] = None,
        port: Optional[Union[str, int]] = None,
        api_key: Optional[str] = None,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Initialize the vector database."""
        self.collection_name = collection_name
        self.embed_model = embed_model or get_embedding_model()
        self.metadata = metadata or {}

        self.client = qdrant_client.QdrantClient(
            location=location, url=url, host=host, port=port, api_key=api_key, **kwargs
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.active_collection = collection_name
        self.type = "qdrant"
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
            self.client.delete_collection(collection_name)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.active_collection = collection_name
        self._index = None
        return self.vector_store

    def get_collection(self, collection_name: str = None) -> Any:
        return self.client.get_collection(collection_name or self.collection_name)

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
        records = self.client.retrieve(
            collection_name=collection_name or self.collection_name,
            ids=ids,
            with_vectors=True,
            with_payload=True,
        )
        docs = []
        for record in records:
            docs.append(
                Document(
                    id=record.id,
                    content=record.payload.get("text", ""),
                    metadata=record.payload.get("metadata", {}),
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
        self.client.delete_collection(collection_name)
        if self.active_collection == collection_name:
            self.active_collection = None

    def get_collections(self) -> Any:
        return self.client.get_collections()

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> QueryResults:
        collection_name = collection_name or self.collection_name
        results = []
        for query in queries:
            try:

                result_points = []
                scroll_result, _ = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="text", match=models.MatchText(text=query)
                            )
                        ]
                    ),
                    limit=n_results,
                    with_payload=True,
                    with_vectors=False,
                )
                result_points = scroll_result
            except Exception as e:
                logger.error(f"Qdrant keyword search failed: {e}")
                result_points = []

            query_result = []
            for point in result_points:
                payload = point.payload or {}
                doc = Document(
                    id=point.id,
                    content=payload.get("text", ""),
                    metadata=payload.get("metadata", {}),
                    embedding=None,
                )
                query_result.append((doc, 1.0))
            results.append(query_result)
        return results
