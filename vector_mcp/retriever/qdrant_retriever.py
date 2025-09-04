#!/usr/bin/python
# coding: utf-8

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb.utils import optional_import_block, require_optional_import
from vector_mcp.vectordb.base import VectorDBFactory

with optional_import_block():
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.vector_stores.pgvecto_rs import PGVectoRsStore
    from sentence_transformers import SentenceTransformer
    import psycopg
    from pgvecto_rs.sdk import PGVectoRs
    from vector_mcp.vectordb.pgvector import PGVectorDB

__all__ = ["QdrantRetriever"]

DEFAULT_COLLECTION_NAME = "memory"
EMPTY_RESPONSE_TEXT = (
    "Empty Response"  # Indicates that the query did not return any results
)
EMPTY_RESPONSE_REPLY = "Sorry, I couldn't find any information on that. If you haven't ingested any documents, please try that."  # Default response for queries without results

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(
    ["pgvector", "psycopg", "llama_index", "sentence_transformers"], "rag"
)
class QdrantRetriever(RAGRetriever):
    """A query engine backed by PGVector that supports document insertion and querying.

    This engine initializes a vector database, builds an index from input documents,
    and allows querying using the chat engine interface.

    Attributes:
        vector_db (PGVectorDB): The PGVector database instance.
        vector_search_engine (PGVectoRsStore): The vector search engine.
        storage_context (StorageContext): The storage context for the vector store.
        index (Optional[VectorStoreIndex]): The index built from the documents.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        connection_string: str,
        database_name: str | None = None,
        embedding_function: Union["BaseEmbedding", Callable[..., Any]] | None = None,  # type: ignore[type-arg]
        embedding_model: Union["BaseEmbedding", str] | None = None,
        collection_name: str | None = None,
    ):
        """Initializes a QdrantRetriever instance.

        Args:
            connection_string (str): Connection string used to connect to PostgreSQL with pgvector.
            database_name (Optional[str]): Name of the PostgreSQL database.
            embedding_function (Optional[Union["BaseEmbedding", Callable[..., Any]]]): Custom embedding function. If None (default),
                defaults to SentenceTransformer encoding.
            embedding_model (Optional[Union["BaseEmbedding", str]]): Embedding model identifier or instance. If None (default),
                "local:all-MiniLM-L6-v2" will be used.
            collection_name (Optional[str]): Name of the PostgreSQL table (collection). If None (default), `DEFAULT_COLLECTION_NAME` will be used.

        Raises:
            ValueError: If no connection string is provided.
        """
        if not connection_string:
            raise ValueError("Connection string is required to connect to PostgreSQL.")

        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.embedding_model = embedding_model or "local:all-MiniLM-L6-v2"  # type: ignore[no-any-unimported]

        # encode is a method of SentenceTransformer, so we need to use a type ignore here.
        self.embedding_function = embedding_function or SentenceTransformer("all-MiniLM-L6-v2").encode  # type: ignore[call-overload]

        self.embed_dim = len(
            self.embedding_function(["The quick brown fox jumps over the lazy dog."])[0]
        )

        # These will be initialized later.
        self.vector_db: PGVectorDB | None = None
        self.vector_search_engine: PGVectoRsStore | None = None  # type: ignore[no-any-unimported]
        self.storage_context: StorageContext | None = None  # type: ignore[no-any-unimported]
        self.index: VectorStoreIndex | None = None  # type: ignore[no-any-unimported]

    def _set_up(self, overwrite: bool) -> None:
        """Sets up the PGVector database, search engine, and storage context.

        This method initializes the vector database using the provided connection details,
        creates a vector search engine instance, and sets the storage context for indexing.

        Args:
            overwrite (bool): Flag indicating whether to overwrite the existing collection.
        """
        logger.info("Setting up the database.")
        self.vector_db: PGVectorDB = VectorDBFactory.create_vector_database(  # type: ignore[assignment, no-redef]
            db_type="pgvector",
            connection_string=self.connection_string,
            dbname=self.database_name,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
            overwrite=overwrite,  # new parameter to control creation behavior
        )
        logger.info("Vector database created.")

        self.vector_search_engine = PGVectoRs(
            db_url=self.connection_string,
            collection_name=self.collection_name,
            dimension=1536,  # Using OpenAI’s text-embedding-ada-002
        )
        logger.info("Vector search engine created.")
        self.storage_context = PGVectoRsStore(client=self.vector_search_engine)

    def _check_existing_collection(self) -> bool:
        """Checks if the specified collection (table) exists in the PostgreSQL database.

        Returns:
            bool: True if the collection exists; False otherwise.
        """
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS (SELECT FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = %s)",
                    (self.collection_name,),
                )
                return cur.fetchone()[0]

    def connect_database(self, *args: Any, **kwargs: Any) -> bool:
        """Connects to the PostgreSQL database and initializes the query index from the existing collection.

        This method verifies the existence of the collection, sets up the database connection,
        builds the vector store index, and pings the PostgreSQL server.

        Returns:
            bool: True if connection is successful; False otherwise.
        """
        try:
            # Check if the target collection exists.
            if not self._check_existing_collection():
                raise ValueError(
                    f"Collection '{self.collection_name}' not found in database '{self.database_name}'. "
                    "Please run init_db to create a new collection."
                )
            # Reinitialize without overwriting the existing collection.
            self._set_up(overwrite=False)

            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_search_engine,  # type: ignore[arg-type]
                storage_context=self.storage_context,
                embed_model=self.embedding_model,
            )

            self.vector_db.client.execute("SELECT 1")  # Simple ping-like query
            logger.info("Connected to PostgreSQL successfully.")
            return True
        except Exception as error:
            logger.error("Failed to connect to PostgreSQL: %s", error)
            return False

    def init_db(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initializes the PostgreSQL database by creating or overwriting the collection and indexing documents.

        This method loads documents from a directory or provided file paths, sets up the database (optionally
        overwriting any existing collection), builds the vector store index, and inserts the documents.

        Args:
            document_directory (Optional[Union[Path, str]]): Directory containing documents to be indexed.
            document_paths (Optional[Sequence[Union[Path, str]]]): List of file paths or URLs for documents.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            bool: True if the database is successfully initialized; False otherwise.
        """
        try:
            # Check if the collection already exists.
            if self._check_existing_collection():
                logger.warning(
                    f"Collection '{self.collection_name}' already exists in database '{self.database_name}'. "
                    "Please use connect_database to connect to the existing collection or use init_db to overwrite it."
                )
            # Set up the database with overwriting.
            self._set_up(overwrite=True)
            self.vector_db.client.execute("SELECT 1")  # Simple ping-like query
            # Gather document paths.
            logger.info("Setting up the database with existing collection.")
            documents = self._load_doc(
                input_dir=document_directory, input_docs=document_paths
            )
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_search_engine,  # type: ignore[arg-type]
                storage_context=self.storage_context,
                embed_model=self.embedding_model,
            )
            for doc in documents:
                self.index.insert(doc)
            logger.info("Database initialized with %d documents.", len(documents))
            return True
        except Exception as e:
            logger.error("Failed to initialize the database: %s", e)
            return False

    def _validate_query_index(self) -> None:
        """Validates that the query index is initialized.

        Raises:
            Exception: If the query index is not initialized.
        """
        if not hasattr(self, "index"):
            raise Exception(
                "Query index is not initialized. Please call init_db or connect_database first."
            )

    def _load_doc(  # type: ignore[no-any-unimported]
        self, input_dir: Path | str | None, input_docs: Sequence[Path | str] | None
    ) -> Sequence["LlamaDocument"]:
        """Loads documents from a directory or a list of file paths.

        Args:
            input_dir (Optional[Union[Path, str]]): Directory from which to load documents.
            input_docs (Optional[Sequence[Union[Path, str]]]): List of document file paths or URLs.

        Returns:
            Sequence[LlamaDocument]: A sequence of loaded LlamaDocument objects.

        Raises:
            ValueError: If the input directory or any specified document file does not exist.
        """
        loaded_documents = []
        if input_dir:
            logger.info("Loading docs from directory: %s", input_dir)
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory not found: {input_dir}")
            loaded_documents.extend(
                SimpleDirectoryReader(input_dir=input_dir).load_data()
            )

        if input_docs:
            for doc in input_docs:
                logger.info("Loading input doc: %s", doc)
                if not os.path.exists(doc):
                    raise ValueError(f"Document file not found: {doc}")
            loaded_documents.extend(SimpleDirectoryReader(input_files=input_docs).load_data())  # type: ignore[arg-type]
        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        return loaded_documents

    def add_documents(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Adds new documents to the existing vector store index.

        This method validates that the index exists, loads documents from the specified directory or file paths,
        and inserts them into the vector store index.

        Args:
            document_directory (Optional[Union[Path, str]]): Directory containing new documents.
            document_paths (Optional[Sequence[Union[Path, str]]]): List of file paths or URLs for new documents.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """
        self._validate_query_index()
        documents = self._load_doc(
            input_dir=document_directory, input_docs=document_paths
        )
        for doc in documents:
            self.index.insert(doc)  # type: ignore[union-attr]

    def query(self, question: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-any-unimported, type-arg]
        """Queries the indexed documents using the provided question.

        This method validates that the query index is initialized, creates a query engine from the vector store index,
        and executes the query. If the response is empty, a default reply is returned.

        Args:
            question (str): The query question.
            args (Any): Additional positional arguments.
            kwargs (Any): Additional keyword arguments.

        Returns:
            Any: The query response as a string, or a default reply if no results are found.
        """
        self._validate_query_index()
        similarity_top_k = kwargs.get("n_results", 10)
        self.retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        response = self.retriever.retrieve(str_or_query_bundle=question)

        if str(response) == EMPTY_RESPONSE_TEXT:
            return EMPTY_RESPONSE_REPLY

        return str(response)

    def get_collection_name(self) -> str:
        """Retrieves the name of the PostgreSQL collection (table).

        Returns:
            str: The collection name.

        Raises:
            ValueError: If the collection name is not set.
        """
        if self.collection_name:
            return self.collection_name
        else:
            raise ValueError("Collection name not set.")


if TYPE_CHECKING:
    from .retriever import RAGQueryEngine

    def _check_implement_protocol(o: QdrantRetriever) -> RAGQueryEngine:
        return o
