#!/usr/bin/python

from typing import Any

from agent_utilities import create_embedding_model
from llama_index.core import (
    Document as LIDocument,
)
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)

from vector_mcp.vectordb.base import Document, ItemID, QueryResults, VectorDB
from vector_mcp.vectordb.db_utils import (
    get_logger,
    optional_import_block,
    require_optional_import,
)

with optional_import_block():
    import qdrant_client
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import models

logger = get_logger(__name__)


@require_optional_import(["qdrant_client", "llama_index"], "qdrant")
class QdrantVectorDB(VectorDB):
    """A vector database that uses Qdrant as the backend via LlamaIndex."""

    def __init__(
        self,
        *,
        location: str | None = None,
        url: str | None = None,
        host: str | int | None = None,
        port: str | int | None = None,
        api_key: str | None = None,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        metadata: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize the vector database."""
        self.collection_name = collection_name
        self.embed_model = embed_model or create_embedding_model()
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
        self, collection_name: str, overwrite: bool = False, _get_or_create: bool = True
    ) -> Any:
        self.collection_name = collection_name

        if overwrite:
            self.client.delete_collection(collection_name)

        # Check if collection exists, if not and _get_or_create is True, create it
        collection_exists = self.client.collection_exists(collection_name)
        if not collection_exists and _get_or_create:
            from qdrant_client.models import Distance, VectorParams

            # Get embedding dimension from the model
            sample_embedding = self.embed_model.get_query_embedding("test")
            vector_size = len(sample_embedding)

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

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

    def get_collection(self, collection_name: str | None = None) -> Any:
        return self.client.get_collection(collection_name or self.collection_name)

    def insert_documents(
        self,
        docs: list[Document],
        collection_name: str | None = None,
        _upsert: bool = False,
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
        collection_name: str | None = None,
        n_results: int = 10,
        distance_threshold: float = -1,
        **kwargs: Any,
    ) -> QueryResults:
        if collection_name:
            self.create_collection(collection_name, _get_or_create=True)

        index = self._get_index()
        results = []
        retriever = index.as_retriever(similarity_top_k=n_results)

        for query in queries:
            nodes = retriever.retrieve(query)
            query_result = []
            for node_match in nodes:
                # node_match.score is already a distance from Qdrant (COSINE distance)
                distance = node_match.score or 0.0
                if distance_threshold >= 0 and distance > distance_threshold:
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
        ids: list[ItemID] | None = None,
        collection_name: str | None = None,
        include=None,
        **kwargs,
    ) -> list[Document]:
        # LlamaIndex stores our original IDs in payload metadata as 'doc_id'
        # Qdrant requires UUID/integer IDs, so we need to search by metadata
        collection_name = collection_name or self.collection_name

        if ids:
            # Get all documents and filter by metadata
            records, _ = self.client.scroll(
                collection_name=collection_name,
                limit=1000,  # Adjust based on expected collection size
                with_payload=True,
                with_vectors=False,
            )
            docs = []
            for record in records:
                payload = record.payload or {}
                doc_id = payload.get("doc_id")
                if doc_id in ids:
                    # Extract metadata from payload (metadata fields are stored at top level)
                    # Filter out LlamaIndex internal fields
                    metadata = {
                        k: v
                        for k, v in payload.items()
                        if not k.startswith("_")
                        and k not in ["doc_id", "ref_doc_id", "document_id"]
                    }
                    # Extract content from _node_content if not available at top level
                    content = payload.get("text", "")
                    if not content and "_node_content" in payload:
                        import json

                        try:
                            node_content = json.loads(payload["_node_content"])
                            content = node_content.get("text", "")
                        except (json.JSONDecodeError, KeyError):
                            content = ""
                    docs.append(
                        Document(
                            id=doc_id,  # Return the original ID
                            content=content,
                            metadata=metadata,
                        )
                    )
            return docs
        else:
            # Return all documents
            records, _ = self.client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            docs = []
            for record in records:
                payload = record.payload or {}
                doc_id = payload.get(
                    "doc_id", str(record.id)
                )  # Use doc_id from metadata or fall back to UUID
                # Extract metadata from payload (metadata fields are stored at top level)
                # Filter out LlamaIndex internal fields
                metadata = {
                    k: v
                    for k, v in payload.items()
                    if not k.startswith("_")
                    and k not in ["doc_id", "ref_doc_id", "document_id"]
                }
                # Extract content from _node_content if not available at top level
                content = payload.get("text", "")
                if not content and "_node_content" in payload:
                    import json

                    try:
                        node_content = json.loads(payload["_node_content"])
                        content = node_content.get("text", "")
                    except (json.JSONDecodeError, KeyError):
                        content = ""
                docs.append(
                    Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
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
        if collection_name:
            self.create_collection(collection_name)
        self.vector_store.delete_nodes(ids)

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name)
        if self.active_collection == collection_name:
            self.active_collection = ""

    def get_collections(self) -> Any:
        return self.client.get_collections()

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
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
