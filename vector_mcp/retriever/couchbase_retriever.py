#!/usr/bin/python
# coding: utf-8

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field
from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb.utils import optional_import_block, require_optional_import
from vector_mcp.vectordb.base import VectorDBFactory

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

with optional_import_block():
    from llama_index.vector_stores.couchbase import CouchbaseVectorStore
    from sentence_transformers import SentenceTransformer
    from vector_mcp.vectordb.couchbase import CouchbaseVectorDB

__all__ = ["CouchbaseRetriever"]

DEFAULT_COLLECTION_NAME = "memory"
EMPTY_RESPONSE_TEXT = "Empty Response"
EMPTY_RESPONSE_REPLY = (
    "Sorry, I couldn't find any information on that. "
    "If you haven't ingested any documents, please try that."
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(["couchbase", "llama_index", "sentence_transformers"], "rag")
class CouchbaseRetriever(RAGRetriever):
    """A query engine backed by Couchbase that supports document insertion and querying.

    This engine initializes a vector database, builds an index from input documents,
    and allows querying using the chat engine interface.

    Attributes:
        vector_db (CouchbaseVectorDB): The Couchbase vector database instance.
        vector_store (CouchbaseVectorStore): The vector store for LlamaIndex.
        storage_context (StorageContext): The storage context for the vector store.
        index (Optional[VectorStoreIndex]): The index built from the documents.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        connection_string: str = Field(
            description="Connection string of Couchbase instance",
            default="couchbase://localhost",
        ),
        username: str = Field(
            description="Username for the Couchbase instance", default="Administrator"
        ),
        password: str = Field(
            description="Password for the Couchbase instance", default="password"
        ),
        bucket_name: str = Field(
            description="Name of the Couchbase bucket", default="vector_db"
        ),
        scope_name: str = Field(
            description="Name of the Couchbase scope", default="_default"
        ),
        collection_name: str = Field(
            description="Name of the Couchbase collection",
            default=DEFAULT_COLLECTION_NAME,
        ),
        index_name: str = Field(
            description="Name of the Couchbase search index", default="vector_index"
        ),
        embedding_function: "EmbeddingFunction[Any] | None" = None,  # type: ignore[type-arg]
    ):
        """Initializes a CouchbaseRetriever instance.

        Args:
            connection_string (str): Connection string used to connect to Couchbase.
            username (str): Username for Couchbase authentication.
            password (str): Password for Couchbase authentication.
            bucket_name (str): Name of the Couchbase bucket.
            scope_name (str): Name of the Couchbase scope.
            collection_name (str): Name of the Couchbase collection (table).
            index_name (str): Name of the Couchbase search index.
            embedding_function (Optional["EmbeddingFunction[Any] | None"]): Custom embedding function. If None (default),
                defaults to SentenceTransformer encoding.

        Raises:
            ValueError: If required connection parameters are not provided.
        """
        if not connection_string:
            raise ValueError("Connection string is required to connect to Couchbase.")
        if not username or not password:
            raise ValueError(
                "Username and password are required for Couchbase authentication."
            )
        if not bucket_name:
            raise ValueError("Bucket name is required for Couchbase connection.")

        self.connection_string = connection_string
        self.username = username
        self.password = password
        self.bucket_name = bucket_name
        self.scope_name = scope_name
        self.collection_name = collection_name
        self.index_name = index_name

        # Initialize embedding function
        self.embedding_function = (
            embedding_function or SentenceTransformer("all-MiniLM-L6-v2").encode
        )
        self.llama_embedding = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.embed_dim = len(
            self.embedding_function(["The quick brown fox jumps over the lazy dog."])[0]
        )

        # These will be initialized later
        self.vector_db: CouchbaseVectorDB | None = None
        self.vector_store: CouchbaseVectorStore | None = None  # type: ignore[no-any-unimported]
        self.storage_context: StorageContext | None = None  # type: ignore[no-any-unimported]
        self.index: VectorStoreIndex | None = None  # type: ignore[no-any-unimported]

    def _set_up(self, overwrite: bool) -> None:
        """Sets up the Couchbase database, vector store, and storage context.

        This method initializes the vector database using the provided connection details,
        creates a collection (with overwrite if specified), creates a vector store instance,
        and sets the storage context for indexing.

        Args:
            overwrite (bool): Flag indicating whether to overwrite the existing collection.
        """
        logger.info("Setting up the Couchbase database.")
        self.vector_db = VectorDBFactory.create_vector_database(
            db_type="couchbase",
            connection_string=self.connection_string,
            username=self.username,
            password=self.password,
            bucket_name=self.bucket_name,
            scope_name=self.scope_name,
            collection_name=self.collection_name,
            index_name=self.index_name,
            embedding_function=self.embedding_function,
        )
        self.vector_db.create_collection(
            collection_name=self.collection_name, overwrite=overwrite
        )
        logger.info("Couchbase vector database created.")

        self.vector_store = CouchbaseVectorStore(
            cluster=self.vector_db.cluster,
            bucket_name=self.bucket_name,
            scope_name=self.scope_name,
            collection_name=self.collection_name,
            index_name=self.index_name,
            embedding_dimension=self.embed_dim,
        )
        logger.info("Couchbase vector store created.")
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def _check_existing_collection(self) -> bool:
        """Checks if the specified collection exists in the Couchbase database.

        Returns:
            bool: True if the collection exists; False otherwise.
        """
        try:
            collection_mgr = self.vector_db.bucket.collections()
            collections = collection_mgr.get_all_collections(self.scope_name)
            return any(
                collection.name == self.collection_name for collection in collections
            )
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def connect_database(
        self, collection_name: str | None = None, *args: Any, **kwargs: Any
    ) -> bool:
        """Connects to the Couchbase database and initializes the query index from the existing collection.

        This method verifies the existence of the collection, sets up the database connection,
        builds the vector store index, and pings the Couchbase server.

        Args:
            collection_name (str, optional): The name of the collection to connect to.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            bool: True if connection is successful; False otherwise.
        """
        if collection_name:
            self.collection_name = collection_name
        try:
            # Check if the target collection exists
            if not self._check_existing_collection():
                raise ValueError(
                    f"Collection '{self.collection_name}' not found in bucket '{self.bucket_name}'. "
                    "Please run initialize_collection to create a new collection."
                )
            # Reinitialize without overwriting the existing collection
            self._set_up(overwrite=False)

            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
                embed_model=self.llama_embedding,
            )

            # Simple ping-like query to verify connection
            self.vector_db.cluster.ping()
            logger.info("Connected to Couchbase successfully.")
            return True
        except Exception as error:
            logger.error(f"Failed to connect to Couchbase: {error}")
            return False

    def initialize_collection(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        overwrite: Optional[bool] | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initializes the Couchbase database by creating or overwriting the collection and indexing documents.

        This method loads documents from a directory or provided file paths, sets up the database (optionally
        overwriting any existing collection), builds the vector store index, and inserts the documents.

        Args:
            document_directory (Optional[Union[Path, str]]): Directory containing documents to be indexed.
            document_paths (Optional[Sequence[Union[Path, str]]]): List of file paths or URLs for documents.
            overwrite (Optional[bool]): Whether to overwrite an existing collection. Defaults to True.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            bool: True if the database is successfully initialized; False otherwise.
        """
        try:
            # Check if the collection already exists
            exists = self._check_existing_collection()
            if exists and not overwrite:
                raise ValueError(
                    f"Collection '{self.collection_name}' already exists in bucket '{self.bucket_name}'. "
                    "Set overwrite=True to overwrite it or use connect_database to connect to the existing collection."
                )
            # Set up the database with overwriting if specified
            self._set_up(overwrite=overwrite)
            self.vector_db.cluster.ping()  # Simple ping-like query
            logger.info("Setting up the database with documents.")
            documents = self._load_doc(
                input_dir=document_directory, input_docs=document_paths
            )
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                embed_model=self.llama_embedding,
            )
            logger.info("Database initialized with %d documents.", len(documents))
            return True
        except Exception as e:
            logger.error(f"Failed to initialize the database: {e}")
            return False

    def _validate_query_index(self) -> None:
        """Validates that the query index is initialized.

        Raises:
            Exception: If the query index is not initialized.
        """
        if not hasattr(self, "index"):
            raise Exception(
                "Query index is not initialized. Please call initialize_collection or connect_database first."
            )

    def _load_doc(
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
            logger.info(f"Loading docs from directory: {input_dir}")
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory not found: {input_dir}")
            loaded_documents.extend(
                SimpleDirectoryReader(input_dir=input_dir).load_data()
            )

        if input_docs:
            for doc in input_docs:
                logger.info(f"Loading input doc: {doc}")
                if not os.path.exists(doc):
                    raise ValueError(f"Document file not found: {doc}")
            loaded_documents.extend(
                SimpleDirectoryReader(input_files=input_docs).load_data()
            )
        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        return loaded_documents

    def add_documents(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Sequence["LlamaDocument"]:
        """Adds new documents to the existing vector store index.

        This method validates that the index exists, loads documents from the specified directory or file paths,
        and inserts them into the vector store index.

        Args:
            document_directory (Optional[Union[Path, str]]): Directory containing new documents.
            document_paths (Optional[Sequence[Union[Path, str]]]): List of file paths or URLs for new documents.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Sequence[LlamaDocument]: The list of documents inserted.
        """
        self._validate_query_index()
        documents = self._load_doc(
            input_dir=document_directory, input_docs=document_paths
        )
        for doc in documents:
            self.index.insert(doc, embed_model=self.llama_embedding)
        return documents

    def query(
        self, question: str, number_results: int, *args: Any, **kwargs: Any
    ) -> str:
        """Queries the indexed documents using the provided question.

        This method validates that the query index is initialized, creates a retriever from the vector store index,
        and executes the query. If the response is empty, a default reply is returned.

        Args:
            question (str): The query question.
            number_results (int): Number of results to return.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            str: The query response as a string, or a default reply if no results are found.
        """
        self._validate_query_index()
        similarity_top_k = kwargs.get("number_results", number_results)
        self.retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        response = self.retriever.retrieve(str_or_query_bundle=question)

        if str(response) == EMPTY_RESPONSE_TEXT:
            return EMPTY_RESPONSE_REPLY

        return str(response)

    def get_collection_name(self) -> str:
        """Retrieves the name of the Couchbase collection.

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

    def _check_implement_protocol(o: CouchbaseRetriever) -> RAGQueryEngine:
        return o
