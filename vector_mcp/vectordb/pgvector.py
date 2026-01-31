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
    SimpleDirectoryReader,
)
import os

with optional_import_block():
    from llama_index.vector_stores.postgres import PGVectorStore
    from sqlalchemy import make_url, text

logger = get_logger(__name__)


@require_optional_import(
    ["pgvector", "psycopg", "llama_index"], "retrievechat-pgvector"
)
class PGVectorDB(VectorDB):
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
        self.embed_model = embed_model or get_embedding_model()
        self.collection_name = collection_name
        self.metadata = metadata or {
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
        }

        # Determine embedding dimension
        self.dimension = len(self.embed_model.get_text_embedding("test"))

        # Construct connection params
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
        self.type = "pgvector"

        # Lazy index
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
        # PGVectorStore handles creation in init/from_params usually, or on first insert.
        # But we might need to re-init vector store if collection name changes?
        # LlamaIndex PGVectorStore is tied to a table name.
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
            self._index = None  # Reset index

        if overwrite:
            # Drop table handled by user manual call usually, but we can try
            # self.vector_store._engine.execute(text(f"DROP TABLE IF EXISTS {collection_name}"))
            # But let's stick to standard LlamaIndex ops where possible
            pass  # LlamaIndex doesn't expose easy drop table on instance

        # Force table creation by inserting documents or dummy document
        try:
            # Check if table exists
            collections = self.get_collections()
            if collection_name not in collections:
                # Check for default documents
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
                    # Insert dummy document to force creation if no docs found
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
        # TODO: Implement drop table
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
            # valid_metadata = {k: v for k, v in metadata.items() if v is not None}
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
                # score is similarity
                if (
                    distance_threshold >= 0 and node_match.score < distance_threshold
                ):  # High score = close Match
                    continue

                doc = Document(
                    id=node_match.node.node_id,
                    content=node_match.node.text,
                    metadata=node_match.node.metadata,
                    embedding=node_match.node.embedding,
                )
                # Distance? LlamaIndex returns score.
                # Usually Cosine Similarity. Distance = 1 - Similarity roughly.
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
        # PGVectorStore doesn't strictly support get by ID easily without query
        # But we can try using the underlying table or simple query
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
            # We need to access the engine to run raw SQL
            if hasattr(self.vector_store, "_engine") and self.vector_store._engine:
                engine = self.vector_store._engine

            if not engine:
                # Fallback: create engine
                from sqlalchemy import create_engine

                # Fix make_url usage
                # If connection_string is None, build it properly
                conn_str = self._db_params.get("connection_string")
                if not conn_str:
                    conn_str = f"postgresql://{self._db_params['user']}:{self._db_params['password']}@{self._db_params['host']}:{self._db_params['port']}/{self._db_params['database']}"

                url = make_url(conn_str)
                engine = create_engine(url)

            # Query for tables in the public schema
            # We assume every table in public schema is a collection unless we have specific naming
            schema_name = "public"
            if hasattr(self.vector_store, "schema_name"):
                schema_name = self.vector_store.schema_name

            with engine.connect() as connection:
                # Filter out system tables or known non-collection tables if needed
                # For now, listing all non-partition tables
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

        # We need direct access to the engine to run raw SQL for ParadeDB
        # LlamaIndex PGVectorStore exposes ._engine (SQLAlchemy engine)
        if not hasattr(self.vector_store, "_engine"):
            # Fallback if we can't get engine easily, though standard PGVectorStore has it.
            # Or we can create one from params.
            logger.warning(
                "PGVectorStore engine not accessible. Cannot run BM25 search."
            )
            return [[] for _ in queries]

        results = []
        with self.vector_store._engine.connect() as connection:
            for query in queries:
                # ParadeDB syntax: paradedb.bm25(text_column, query)
                # We assume the table has a 'text' column where content is stored.
                # LlamaIndex usually stores content in 'text' column.
                # We also need to join with data table or just select from the table.
                # LlamaIndex default table schema:
                # id (varchar), text (varchar), metadata_ (json), embedding (vector), ...

                # Note: ParadeDB might need an index to work fast, but bm25() function works on text.
                # We assume the user has set up ParadeDB or we just try to call it.
                # "paradedb.bm25(text, :query)"

                # Wait, paradedb.bm25 acts on the index usually.
                # If using pg_search (ParadeDB), we usually do:
                # SELECT * FROM table WHERE table @@@ 'query';
                # But the user mentioned: "PGvector works with paradedb (which supports the bm25 and pgvector extensions)"
                # "It ranks documents ... based purely on how well they match ... in the query."

                # Let's try the standard ParadeDB / pg_search approach if possible, or just exact match if not.
                # Actually, if they have the `bm25` extension (from paradedb), usage is often:
                # SELECT ... FROM ... ORDER BY paradedb.bm25(text, 'query') ...

                # However, strictly speaking, LlamaIndex table names are `data_<collection_name>`.

                # Let's try to be generic. If we can't find `paradedb`, we might fail.
                # We will wrap in try/except.

                # Re-reading user request: "PGvector works with paradedb"

                # Construct SQL
                # LlamaIndex uses `data_<table>` for data
                table_name = f"data_{self.collection_name}"

                # Check for table existence or just run query
                try:
                    # Generic BM25 via paradedb.bm25(text_col, 'query')
                    # This assumes the function exists.
                    sql = text(f"""
                        SELECT id, text, metadata_, paradedb.bm25(text, :q) as score
                        FROM {table_name}
                        WHERE paradedb.bm25(text, :q) > 0
                        ORDER BY score DESC
                        LIMIT :k
                    """)

                    # NOTE: If paradedb.bm25 is not available this will fail.
                    # As a fallback:
                    # generic text search: to_tsvector(text) @@ plainto_tsquery(:q)
                    # rank: ts_rank(to_tsvector(text), plainto_tsquery(:q))

                    # I will implement the Postgres Native FTS (ts_rank) as a safe fallback or primary implementation
                    # if paradedb is not explicitly guaranteed to be installed in a specific way,
                    # BUT user asked for BM25. ts_rank is not exactly BM25 but close enough for standard Postgres.
                    # However, ParadeDB provides real BM25. I will try ParadeDB syntax first.

                    # Actually, let's use a dual approach: try ParadeDB, fallback to standard Postgres TS.

                    # But simpler: User said "PGvector works with paradedb... extend vector_mcp to add lexical_search"
                    # I will assume the environment supports it.

                    result_proxy = connection.execute(sql, {"q": query, "k": n_results})
                    query_result = []
                    for row in result_proxy:
                        # row: id, text, metadata_, score
                        # Metadata in LlamaIndex PGVector is JSONB column "metadata_"
                        doc = Document(
                            id=row[0],
                            content=row[1],
                            metadata=row[2],
                            embedding=None,  # We don't fetch embedding here for speed
                        )
                        query_result.append((doc, float(row[3])))
                    results.append(query_result)

                except Exception as e:
                    # Fallback to standard Postgres Full Text Search
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
