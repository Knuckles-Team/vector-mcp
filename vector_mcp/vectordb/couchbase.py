#!/usr/bin/python
# coding: utf-8
from typing import Any, Optional, Union

from vector_mcp.vectordb.base import Document, ItemID, QueryResults, VectorDB
from vector_mcp.vectordb.db_utils import (
    get_logger,
    optional_import_block,
    require_optional_import,
)

from agent_utilities import create_embedding_model

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document as LIDocument,
)

with optional_import_block():
    from couchbase.cluster import Cluster
    from couchbase.auth import PasswordAuthenticator
    from couchbase.options import ClusterOptions
    from llama_index.vector_stores.couchbase import CouchbaseVectorStore

logger = get_logger(__name__)


@require_optional_import(["couchbase", "llama_index"], "retrievechat-couchbase")
class CouchbaseVectorDB(VectorDB):
    """A vector database that uses Couchbase as the backend via LlamaIndex."""

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
        self.embed_model = embed_model or create_embedding_model()
        self.metadata = metadata or {}

        if not connection_string:
            connection_string = f"couchbase://{host or 'localhost'}"

        self.cluster = Cluster(
            connection_string, ClusterOptions(PasswordAuthenticator(username, password))
        )
        self.bucket_name = dbname or "default"
        self.scope_name = kwargs.get("scope_name", "_default")

        self.vector_store = CouchbaseVectorStore(
            cluster=self.cluster,
            bucket_name=self.bucket_name,
            scope_name=self.scope_name,
            collection_name=self.collection_name,
            **kwargs,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.active_collection = collection_name
        self.type = "couchbase"
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
        self.vector_store = CouchbaseVectorStore(
            cluster=self.cluster,
            bucket_name=self.bucket_name,
            scope_name=self.scope_name,
            collection_name=self.collection_name,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.active_collection = collection_name
        self._index = None

        if overwrite:
            pass
        return self.vector_store

    def get_collection(self, collection_name: str = None) -> Any:
        return self.vector_store

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
        bucket = self.cluster.bucket(self.bucket_name)
        scope = bucket.scope(self.scope_name)
        coll = scope.collection(collection_name or self.collection_name)

        docs = []
        for _id in ids:
            try:
                res = coll.get(_id)
                content = res.content_as[dict]
                docs.append(
                    Document(
                        id=_id,
                        content=content.get("text", ""),
                        metadata=content.get("metadata", {}),
                    )
                )
            except Exception:
                continue
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
        pass

    def get_collections(self) -> Any:
        return []

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> QueryResults:
        collection_name = collection_name or self.collection_name

        results = []
        for query_text in queries:
            try:
                index_name = collection_name

                from couchbase.search import SearchOptions, MatchQuery

                search_result = self.cluster.search_query(
                    index_name, MatchQuery(query_text), SearchOptions(limit=n_results)
                )

                query_result = []
                for row in search_result.rows():

                    bucket = self.cluster.bucket(self.bucket_name)
                    scope = bucket.scope(self.scope_name)
                    coll = scope.collection(collection_name)

                    try:
                        doc_kv = coll.get(row.id)
                        content = doc_kv.content_as[dict]
                        doc = Document(
                            id=row.id,
                            content=content.get("text", "")
                            or content.get("content", ""),
                            metadata=content.get("metadata", {}),
                            embedding=None,
                        )
                        query_result.append((doc, row.score))
                    except Exception:
                        pass
                results.append(query_result)

            except Exception as e:
                logger.error(f"Couchbase search failed: {e}")
                results.append([])

        return results
