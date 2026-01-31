#!/usr/bin/python
# coding: utf-8
import os
from typing import Any, Optional

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
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

logger = get_logger(__name__)


@require_optional_import(["chromadb", "llama_index"], "retrievechat")
class ChromaVectorDB(VectorDB):
    """A vector database that uses ChromaDB as the backend via LlamaIndex."""

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        path: Optional[str] = None,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Initialize the vector database.

        Args:
           client: chromadb.Client | Existing client
           path: str | Path for persistent client
           collection_name: str | Collection name
        """
        self.embed_model = embed_model or get_embedding_model()
        self.active_collection = None
        self.type = ""
        self.collection_name = collection_name
        self.metadata = metadata or {}

        if client:
            self.client = client
        else:
            if path:
                self.path = path
                self.client = chromadb.PersistentClient(path=self.path)
            else:
                self.path = os.path.expanduser("~/Documents/ChromaDB")
                self.client = chromadb.PersistentClient(path=self.path)

        self.chroma_collection = self.client.get_or_create_collection(
            self.collection_name
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.active_collection = collection_name
        self.type = "chroma"
        self._index = None

    def _get_index(self) -> VectorStoreIndex:
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
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

        self.chroma_collection = self.client.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.active_collection = collection_name
        self._index = None
        return self.chroma_collection

    def get_collection(self, collection_name: str = None) -> Any:
        name = collection_name or self.collection_name
        return self.client.get_collection(name)

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
        collection = self.get_collection(collection_name)
        # Chroma native get
        results = collection.get(ids=ids, include=include or ["metadatas", "documents"])
        # parse results which are dict of lists
        docs = []
        if results and results["ids"]:
            for i, _id in enumerate(results["ids"]):
                docs.append(
                    Document(
                        id=_id,
                        content=results["documents"][i] if results["documents"] else "",
                        metadata=(
                            results["metadatas"][i] if results["metadatas"] else {}
                        ),
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
        return self.client.list_collections()

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> QueryResults:
        collection = self.get_collection(collection_name)
        results = []
        for query in queries:
            # ChromaDB v0.4+ supports `where_document={"$contains": "search_term"}`
            # This is a substring match, not true BM25, but it's the closest "keyword search"
            # we can get without managing our own index or using 3rd party tools.
            # However, `query` method always expects embeddings or does embedding on `query_texts`.
            # If we just want keyword match, we can use `get` with `where_document`.
            # But `get` doesn't rank by relevance (it just returns matches).

            # Try to use `query` but we need to supply embeddings or text.
            # If we supply text, it does vector search + can filter.
            # But user wants BM25.

            # Alternative: pure "keyword" search using `get` and `where_document`.
            query_res = collection.get(
                where_document={"$contains": query},
                include=["documents", "metadatas"],
                limit=n_results,
            )
            # Result is dict: {'ids': [], 'documents': [], 'metadatas': []}

            # If we want some "score", maybe we can't get it easily.
            # We will just return 1.0 or attempt a simple count? No, just 1.0.

            query_result = []
            if query_res and query_res["ids"]:
                for i, _id in enumerate(query_res["ids"]):
                    doc = Document(
                        id=_id,
                        content=(
                            query_res["documents"][i] if query_res["documents"] else ""
                        ),
                        metadata=(
                            query_res["metadatas"][i] if query_res["metadatas"] else {}
                        ),
                        embedding=None,
                    )
                    query_result.append((doc, 1.0))
            results.append(query_result)
        return results
