#!/usr/bin/python

import json
import math
from typing import Any

import requests
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
    from couchbase.auth import PasswordAuthenticator
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions

logger = get_logger(__name__)


@require_optional_import(["couchbase", "llama_index"], "couchbase")
class CouchbaseVectorDB(VectorDB):
    """A vector database that uses Couchbase as the backend via simple client approach."""

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

        if not connection_string:
            connection_string = f"couchbase://{host or 'localhost'}"

        self.connection_string = connection_string
        self.username = username
        self.password = password
        self.bucket_name = dbname or "default"
        self.scope_name = kwargs.get("scope_name", "_default")
        self.host = host or "localhost"
        self.port = port or 8091
        self.query_port = kwargs.get("query_port", 8093)  # N1QL query service port

        # Try simple client approach first (bypass LlamaIndex authentication issues)
        self._use_simple_client = True
        self.cluster = None
        self.vector_store = None
        self.storage_context = None
        self.active_collection = collection_name
        self.type = "couchbase"
        self._index = None

        # Try to connect with simple client approach
        self._connect_simple_client()

    def _connect_simple_client(self):
        """Try to connect using simple client approach."""
        try:
            # Try connection without authentication first (for local testing)
            try:
                self.cluster = Cluster(self.connection_string)
                if self.cluster:
                    self.cluster.ping()
                    logger.info("Connected to Couchbase without authentication")
                    return
            except Exception as e:
                logger.info(
                    f"Connection without auth failed: {e}, trying with authentication"
                )

            # Try with authentication
            if self.username and self.password:
                try:
                    options = ClusterOptions(
                        PasswordAuthenticator(self.username, self.password)
                    )
                    self.cluster = Cluster(self.connection_string, options)
                    if self.cluster:
                        self.cluster.ping()
                        logger.info("Connected to Couchbase with authentication")
                        return
                except Exception as e:
                    logger.warning(f"Connection with auth failed: {e}")

            # If SDK connection fails, we'll use REST API for operations
            logger.warning("SDK connection failed, will use REST API for operations")
            self.cluster = None

        except Exception as e:
            logger.error(f"Couchbase connection error: {e}")
            self.cluster = None

    def _get_collection(self, collection_name: str):
        """Get Couchbase collection for a given name."""
        if self.cluster is None:
            return None

        try:
            bucket = self.cluster.bucket(self.bucket_name)
            scope = bucket.scope(self.scope_name)
            return scope.collection(collection_name)
        except Exception as e:
            logger.error(f"Error getting collection {collection_name}: {e}")
            return None

    def _get_index(self) -> "VectorStoreIndex":
        # For simple client approach, we'll implement basic operations without LlamaIndex index
        return None

    def create_collection(
        self, collection_name: str, overwrite: bool = False, _get_or_create: bool = True
    ) -> Any:
        self.collection_name = collection_name

        # Try to create collection via REST API if SDK is not available
        if self.cluster is None:
            self._create_collection_via_rest(collection_name, overwrite)
        else:
            # Try to create collection via SDK
            try:
                bucket = self.cluster.bucket(self.bucket_name)
                scope = bucket.scope(self.scope_name)
                if overwrite:
                    try:
                        scope.collection_drop(collection_name)
                    except Exception:
                        pass
                try:
                    scope.collection_create(collection_name)
                except Exception:
                    pass  # Collection might already exist
            except Exception as e:
                logger.error(f"Error creating collection via SDK: {e}")
                # Fall back to REST API
                self._create_collection_via_rest(collection_name, overwrite)

        self.active_collection = collection_name
        return self._get_collection(collection_name)

    def _create_collection_via_rest(
        self, collection_name: str, overwrite: bool = False
    ):
        """Create collection via Couchbase REST API."""
        try:
            # Check if collection already exists
            scopes_url = f"http://{self.host}:{self.port}/pools/default/buckets/{self.bucket_name}/scopes"
            auth = (
                (self.username, self.password)
                if self.username and self.password
                else None
            )

            response = requests.get(scopes_url, auth=auth)
            if response.status_code == 200:
                result = response.json()
                for scope in result.get("scopes", []):
                    if scope.get("name") == self.scope_name:
                        for coll in scope.get("collections", []):
                            if coll.get("name") == collection_name:
                                if overwrite:
                                    self._delete_collection_via_rest(collection_name)
                                else:
                                    logger.info(
                                        f"Collection {collection_name} already exists"
                                    )
                                    return

            # Try to create collection using N1QL
            query_url = f"http://{self.host}:{self.query_port}/query/service"
            query = f"CREATE COLLECTION IF NOT EXISTS `{self.bucket_name}`.`{self.scope_name}`.`{collection_name}`"
            response = requests.post(query_url, json={"statement": query}, auth=auth)

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    logger.info(f"Collection {collection_name} created via N1QL")
                    return

            # Fallback to REST API
            url = f"http://{self.host}:{self.port}/pools/default/buckets/{self.bucket_name}/scopes/{self.scope_name}/collections"
            data = {"name": collection_name}
            response = requests.post(url, json=data, auth=auth)
            if response.status_code in [200, 201, 202]:
                logger.info(f"Collection {collection_name} created via REST API")
            else:
                logger.warning(
                    f"Failed to create collection via REST API: {response.status_code}"
                )
        except Exception as e:
            logger.error(f"Error creating collection via REST API: {e}")

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

            if collection:
                try:
                    if _upsert:
                        collection.upsert(doc_id, document)
                    else:
                        collection.insert(doc_id, document)
                except Exception as e:
                    logger.error(f"Error inserting document {doc_id}: {e}")
            else:
                # Fall back to REST API
                self._insert_document_via_rest(
                    str(doc_id) if doc_id else "",
                    document,
                    collection_name or self.collection_name,
                    _upsert,
                )

    def _insert_document_via_rest(
        self, doc_id: str, document: dict, collection_name: str, upsert: bool = False
    ):
        """Insert document via Couchbase REST API using N1QL."""
        try:
            # Use N1QL for document insertion
            url = f"http://{self.host}:{self.query_port}/query/service"
            auth = (
                (self.username, self.password)
                if self.username and self.password
                else None
            )

            # Convert document to N1QL format
            text = document.get("text", "").replace("'", "''")
            metadata = document.get("metadata", {})
            embedding = document.get("embedding", [])

            # Create N1QL query for upsert
            query = f"""
            UPSERT INTO `{self.bucket_name}`.`{self.scope_name}`.`{collection_name}`
            (KEY, id, text, metadata, embedding)
            VALUES ('{doc_id}', '{doc_id}', '{text}', {json.dumps(metadata).replace('"', "'")}, {json.dumps(embedding).replace('"', "'")})
            """

            response = requests.post(url, json={"statement": query}, auth=auth)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    logger.info(f"Document {doc_id} inserted via N1QL")
                    return

            # Try simple INSERT as fallback
            query = f"""
            INSERT INTO `{self.bucket_name}`.`{self.scope_name}`.`{collection_name}`
            (KEY, id, text, metadata, embedding)
            VALUES ('{doc_id}', '{doc_id}', '{text}', {json.dumps(metadata).replace('"', "'")}, {json.dumps(embedding).replace('"', "'")})
            """
            response = requests.post(url, json={"statement": query}, auth=auth)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    logger.info(f"Document {doc_id} inserted via N1QL INSERT")
                    return

            logger.warning(
                f"Failed to insert document via N1QL: {response.status_code}"
            )
        except Exception as e:
            logger.error(f"Error inserting document via N1QL: {e}")

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

        for query in queries:
            query_embedding = self.embed_model.get_query_embedding(query)
            query_result = []

            # Get all documents and calculate cosine similarity
            documents = self._get_all_documents(
                collection, collection_name or self.collection_name
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

    def _get_all_documents(self, collection, collection_name: str) -> list:
        """Get all documents from collection."""
        documents = []

        if collection:
            try:
                # Use SDK to get all documents
                # Note: Couchbase doesn't have a simple "get all" operation
                # We'll need to use a query or iterate through keys
                # For now, return empty list and implement proper iteration later
                pass
            except Exception as e:
                logger.error(f"Error getting documents via SDK: {e}")

        # Fall back to REST API to get documents
        documents = self._get_documents_via_rest(collection_name)
        return documents

    def _get_documents_via_rest(self, collection_name: str) -> list:
        """Get all documents via Couchbase REST API."""
        try:
            # Use N1QL query to get all documents
            url = f"http://{self.host}:{self.query_port}/query/service"
            auth = (
                (self.username, self.password)
                if self.username and self.password
                else None
            )

            query = f"SELECT id, text, metadata, embedding FROM `{self.bucket_name}`.`{self.scope_name}`.`{collection_name}`"
            response = requests.post(url, json={"statement": query}, auth=auth)

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result.get("results", [])
            else:
                logger.warning(
                    f"Failed to get documents via REST API: {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error getting documents via REST API: {e}")

        return []

    def get_documents_by_ids(
        self,
        ids: list[ItemID] | None = None,
        collection_name: str | None = None,
        include=None,
        **kwargs,
    ) -> list[Document]:
        collection = self._get_collection(collection_name or self.collection_name)
        docs = []

        if not ids:
            return []

        if collection:
            for _id in ids:
                try:
                    res = collection.get(_id)
                    content = res.content_as[dict]
                    docs.append(
                        Document(
                            id=_id,
                            content=content.get("text", ""),
                            metadata=content.get("metadata", {}),
                        )
                    )
                except Exception as e:
                    logger.error(f"Error getting document {_id}: {e}")
                    # Try REST API as fallback
                    doc = self._get_document_via_rest(
                        str(_id), collection_name or self.collection_name
                    )
                    if doc:
                        docs.append(doc)
        else:
            # Use REST API
            for _id in ids:
                doc = self._get_document_via_rest(
                    str(_id), collection_name or self.collection_name
                )
                if doc:
                    docs.append(doc)

        return docs

    def _get_document_via_rest(
        self, doc_id: str, collection_name: str
    ) -> Document | None:
        """Get document via Couchbase REST API using N1QL."""
        try:
            url = f"http://{self.host}:{self.query_port}/query/service"
            auth = (
                (self.username, self.password)
                if self.username and self.password
                else None
            )

            # Use N1QL to get document
            query = f"SELECT id, text, metadata FROM `{self.bucket_name}`.`{self.scope_name}`.`{collection_name}` WHERE META().id = '{doc_id}'"
            response = requests.post(url, json={"statement": query}, auth=auth)

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    results = result.get("results", [])
                    if results:
                        row = results[0]
                        return Document(
                            id=row.get("id"),
                            content=row.get("text", ""),
                            metadata=row.get("metadata", {}),
                        )
        except Exception as e:
            logger.error(f"Error getting document {doc_id} via N1QL: {e}")

        return None

    def update_documents(
        self, docs: list[Document], collection_name: str | None = None, **kwargs
    ) -> None:
        self.insert_documents(docs, collection_name, upsert=True, **kwargs)

    def delete_documents(
        self, ids: list[ItemID], collection_name: str | None = None, **kwargs
    ) -> None:
        collection = self._get_collection(collection_name or self.collection_name)

        if collection:
            for _id in ids:
                try:
                    collection.remove(_id)
                except Exception as e:
                    logger.error(f"Error deleting document {_id}: {e}")
                    # Try REST API as fallback
                    self._delete_document_via_rest(
                        str(_id), collection_name or self.collection_name
                    )
        else:
            # Use REST API
            for _id in ids:
                self._delete_document_via_rest(
                    str(_id), collection_name or self.collection_name
                )

    def _delete_document_via_rest(self, doc_id: str, collection_name: str):
        """Delete document via Couchbase REST API using N1QL."""
        try:
            url = f"http://{self.host}:{self.query_port}/query/service"
            auth = (
                (self.username, self.password)
                if self.username and self.password
                else None
            )

            # Use N1QL to delete document
            query = f"DELETE FROM `{self.bucket_name}`.`{self.scope_name}`.`{collection_name}` WHERE META().id = '{doc_id}'"
            response = requests.post(url, json={"statement": query}, auth=auth)

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    logger.info(f"Document {doc_id} deleted via N1QL")
                    return

            logger.warning(
                f"Failed to delete document via N1QL: {response.status_code}"
            )
        except Exception as e:
            logger.error(f"Error deleting document via N1QL: {e}")

    def delete_collection(self, collection_name: str) -> None:
        collection = self._get_collection(collection_name)
        if collection and self.cluster:
            try:
                bucket = self.cluster.bucket(self.bucket_name)
                scope = bucket.scope(self.scope_name)
                scope.collection_drop(collection_name)
            except Exception as e:
                logger.error(f"Error deleting collection via SDK: {e}")
                # Try REST API
                self._delete_collection_via_rest(collection_name)
        else:
            self._delete_collection_via_rest(collection_name)

        if self.active_collection == collection_name:
            self.active_collection = ""

    def _delete_collection_via_rest(self, collection_name: str):
        """Delete collection via Couchbase REST API."""
        try:
            # Try to delete collection using N1QL
            query_url = f"http://{self.host}:{self.port}/query/service"
            auth = (
                (self.username, self.password)
                if self.username and self.password
                else None
            )

            query = f"DROP COLLECTION IF EXISTS `{self.bucket_name}`.`{self.scope_name}`.`{collection_name}`"
            response = requests.post(query_url, json={"statement": query}, auth=auth)

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    logger.info(f"Collection {collection_name} deleted via N1QL")
                    return

            # Fallback to REST API
            url = f"http://{self.host}:{self.port}/pools/default/buckets/{self.bucket_name}/scopes/{self.scope_name}/collections/{collection_name}"
            response = requests.delete(url, auth=auth)
            if response.status_code in [200, 202]:
                logger.info(f"Collection {collection_name} deleted via REST API")
            else:
                logger.warning(
                    f"Failed to delete collection via REST API: {response.status_code}"
                )
        except Exception as e:
            logger.error(f"Error deleting collection via REST API: {e}")

    def get_collections(self) -> Any:
        try:
            if self.cluster:
                bucket = self.cluster.bucket(self.bucket_name)
                scope = bucket.scope(self.scope_name)
                collections = scope.collections()
                return list(collections.keys())
            else:
                # Use REST API
                return self._get_collections_via_rest()
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return self._get_collections_via_rest()

    def _get_collections_via_rest(self) -> list:
        """Get collections via Couchbase REST API."""
        try:
            url = f"http://{self.host}:{self.port}/pools/default/buckets/{self.bucket_name}/scopes/{self.scope_name}/collections"
            auth = (
                (self.username, self.password)
                if self.username and self.password
                else None
            )

            response = requests.get(url, auth=auth)
            if response.status_code == 200:
                result = response.json()
                return [
                    col.get("name")
                    for col in result.get("scopes", [{}])[0].get("collections", [])
                ]
        except Exception as e:
            logger.error(f"Error getting collections via REST API: {e}")

        return []

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> QueryResults:
        collection_name = collection_name or self.collection_name

        results = []
        for query_text in queries:
            try:
                # Try to use SDK for search if available
                if self.cluster:
                    try:
                        from couchbase.search import MatchQuery, SearchOptions

                        search_result = self.cluster.search_query(
                            collection_name,
                            MatchQuery(query_text),
                            SearchOptions(limit=n_results),
                        )

                        query_result = []
                        for row in search_result.rows():
                            doc = self._get_document_via_rest(row.id, collection_name)
                            if doc:
                                query_result.append((doc, row.score))
                        results.append(query_result)
                        continue
                    except Exception as e:
                        logger.error(f"SDK search failed: {e}")

                # Fall back to simple text search via N1QL
                query_result = self._lexical_search_via_n1ql(
                    query_text, collection_name, n_results
                )
                results.append(query_result)

            except Exception as e:
                logger.error(f"Couchbase search failed: {e}")
                results.append([])

        return results

    def _lexical_search_via_n1ql(
        self, query_text: str, collection_name: str, n_results: int
    ) -> list:
        """Perform lexical search using N1QL query."""
        try:
            url = f"http://{self.host}:{self.query_port}/query/service"
            auth = (
                (self.username, self.password)
                if self.username and self.password
                else None
            )

            # Simple LIKE query for text search
            query = f"SELECT id, text, metadata FROM `{self.bucket_name}`.`{self.scope_name}`.`{collection_name}` WHERE text LIKE '%{query_text}%' LIMIT {n_results}"
            response = requests.post(url, json={"statement": query}, auth=auth)

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    query_result = []
                    for row in result.get("results", []):
                        doc = Document(
                            id=row.get("id"),
                            content=row.get("text", ""),
                            metadata=row.get("metadata", {}),
                            embedding=None,
                        )
                        query_result.append(
                            (doc, 1.0)
                        )  # Use 1.0 as score for simple text search
                    return query_result

        except Exception as e:
            logger.error(f"Error in N1QL search: {e}")

        return []
