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
    SimpleDirectoryReader,
)
import os

with optional_import_block():
    from llama_index.vector_stores.postgres import PGVectorStore
    from sqlalchemy import make_url, text

logger = get_logger(__name__)


@require_optional_import(
    ["postgres", "psycopg", "llama_index"], "retrievechat-postgres"
)
class PostgreSQL(VectorDB):
    """A vector database that uses PGVector as the backend via LlamaIndex."""

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
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the vector database with LlamaIndex PGVectorStore.

        Args:
            connection_string: str | Full connection string
            host, port, dbname, username, password: Connection details if no connection_string
            embed_model: BaseEmbedding | Custom embedding model
            collection_name: str | Name of the table/collection
            metadata: dict | HNSW index params
        """
        self.embed_model = embed_model or create_embedding_model()
        self.collection_name = collection_name
        self.metadata = metadata or {
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
        }

        self.dimension = len(self.embed_model.get_text_embedding("test"))

        if connection_string:
            url = make_url(connection_string)
            self._db_params = {
                "database": url.database,
                "host": url.host,
                "password": url.password,
                "port": url.port or 5432,
                "user": url.username,
            }
        else:
            self._db_params = {
                "database": dbname,
                "host": str(host),
                "password": password,
                "port": int(port) if port else 5432,
                "user": username,
            }

        self.vector_store = PGVectorStore.from_params(
            **self._db_params,
            table_name=self.collection_name,
            embed_dim=self.dimension,
            hnsw_kwargs=self.metadata,
            **kwargs,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.active_collection = collection_name
        self.type = "postgres"

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
        if collection_name != self.vector_store.table_name:
            self.vector_store = PGVectorStore.from_params(
                **self._db_params,
                table_name=collection_name,
                embed_dim=self.dimension,
                hnsw_kwargs=self.metadata,
            )
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self._index = None

        if overwrite:
            pass

        try:
            collections = self.get_collections()
            if collection_name not in collections:
                doc_dir = os.environ.get(
                    "DOCUMENT_DIRECTORY", os.path.normpath("/documents")
                )
                loaded_docs = []
                if os.path.exists(doc_dir) and os.listdir(doc_dir):
                    try:
                        logger.info(
                            f"Loading documents from {doc_dir} for new collection {collection_name}"
                        )
                        reader = SimpleDirectoryReader(input_dir=doc_dir)
                        loaded_docs = reader.load_data()
                    except Exception as e:
                        logger.warning(f"Failed to load documents from {doc_dir}: {e}")

                if loaded_docs:
                    index = self._get_index()
                    for doc in loaded_docs:
                        index.insert(doc)
                    logger.info(
                        f"Initialized collection {collection_name} with {len(loaded_docs)} documents."
                    )
                else:
                    dummy_doc = LIDocument(
                        text="initialization", doc_id="init_doc", metadata={}
                    )
                    index = self._get_index()
                    index.insert(dummy_doc)
                    index.delete_ref_doc("init_doc", delete_from_docstore=True)
                    logger.info(f"Initialized empty collection {collection_name}.")
        except Exception as e:
            logger.warning(f"Failed to force create table: {e}")

        self.active_collection = collection_name
        return self.vector_store

    def get_collection(self, collection_name: str = None) -> Any:
        name = collection_name or self.active_collection
        if name != self.collection_name:
            self.create_collection(name)
        return self.vector_store

    def delete_collection(self, collection_name: str) -> Any:
        pass

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
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return []

    def update_documents(
        self, docs: list[Document], collection_name: str = None, **kwargs
    ) -> None:
        self.insert_documents(docs, collection_name, upsert=True)

    def delete_documents(
        self, ids: list[ItemID], collection_name: str = None, **kwargs
    ) -> None:
        if collection_name:
            self.create_collection(collection_name)
        self.vector_store.delete_nodes(ids)

    def get_collections(self) -> Any:
        try:
            engine = None
            if hasattr(self.vector_store, "_engine") and self.vector_store._engine:
                engine = self.vector_store._engine

            if not engine:
                from sqlalchemy import create_engine

                conn_str = self._db_params.get("connection_string")
                if not conn_str:
                    conn_str = f"postgresql://{self._db_params['user']}:{self._db_params['password']}@{self._db_params['host']}:{self._db_params['port']}/{self._db_params['database']}"

                url = make_url(conn_str)
                engine = create_engine(url)

            schema_name = "public"
            if hasattr(self.vector_store, "schema_name"):
                schema_name = self.vector_store.schema_name

            with engine.connect() as connection:
                query = text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = :schema
                    AND table_type = 'BASE TABLE'
                """)
                result = connection.execute(query, {"schema": schema_name})
                tables = []
                for row in result:
                    name = row[0]
                    if name.startswith("data_"):
                        name = name[5:]
                    tables.append(name)
                return tables

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> QueryResults:
        if collection_name:
            self.create_collection(collection_name)

        if not hasattr(self.vector_store, "_engine"):
            logger.warning(
                "PGVectorStore engine not accessible. Cannot run BM25 search."
            )
            return [[] for _ in queries]

        results = []
        with self.vector_store._engine.connect() as connection:
            for query in queries:

                table_name = f"data_{self.collection_name}"

                try:
                    sql = text(f"""
                        SELECT id, text, metadata_, paradedb.bm25(text, :q) as score
                        FROM {table_name}
                        WHERE paradedb.bm25(text, :q) > 0
                        ORDER BY score DESC
                        LIMIT :k
                    """)

                    result_proxy = connection.execute(sql, {"q": query, "k": n_results})
                    query_result = []
                    for row in result_proxy:
                        doc = Document(
                            id=row[0],
                            content=row[1],
                            metadata=row[2],
                            embedding=None,
                        )
                        query_result.append((doc, float(row[3])))
                    results.append(query_result)

                except Exception as e:
                    logger.warning(
                        f"ParadeDB BM25 failed, falling back to Postgres native FTS: {e}"
                    )
                    try:
                        sql_fallback = text(f"""
                            SELECT id, text, metadata_, ts_rank(to_tsvector('english', text), plainto_tsquery('english', :q)) as score
                            FROM {table_name}
                            WHERE to_tsvector('english', text) @@ plainto_tsquery('english', :q)
                            ORDER BY score DESC
                            LIMIT :k
                        """)
                        result_proxy = connection.execute(
                            sql_fallback, {"q": query, "k": n_results}
                        )
                        query_result = []
                        for row in result_proxy:
                            doc = Document(
                                id=row[0],
                                content=row[1],
                                metadata=row[2],
                                embedding=None,
                            )
                            query_result.append((doc, float(row[3])))
                        results.append(query_result)
                    except Exception as e2:
                        logger.error(f"Text search failed: {e2}")
                        results.append([])

        return results
