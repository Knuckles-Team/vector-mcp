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
    from qdrant_client import QdrantClient
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from vector_mcp.vectordb.qdrant import QdrantVectorDB, FastEmbedEmbeddingFunction

__all__ = ["QdrantRetriever"]

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


@require_optional_import(["qdrant_client", "llama_index", "fastembed"], "rag")
class QdrantRetriever(RAGRetriever):
    """A query engine backed by Qdrant that supports document insertion and querying.

    This engine initializes a vector database, builds an index from input documents,
    and allows querying using the chat engine interface.

    Attributes:
        vector_db (QdrantVectorDB): The Qdrant vector database instance.
        vector_store (QdrantVectorStore): The vector store for LlamaIndex.
        storage_context (StorageContext): The storage context for the vector store.
        index (Optional[VectorStoreIndex]): The index built from the documents.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        location: str = Field(
            description="Location of Qdrant instance (e.g., ':memory:', 'localhost:6333', or URL)",
            default=":memory:",
        ),
        collection_name: str = Field(
            description="Name of the Qdrant collection", default=DEFAULT_COLLECTION_NAME
        ),
        embedding_function: "EmbeddingFunction[Any] | None" = None,  # type: ignore[type-arg]
        content_payload_key: str = Field(
            description="Key for content payload in Qdrant", default="_content"
        ),
        metadata_payload_key: str = Field(
            description="Key for metadata payload in Qdrant", default="_metadata"
        ),
        collection_options: dict = Field(
            description="Options for creating the Qdrant collection",
            default=None,
        ),
    ):
        """Initializes a QdrantRetriever instance.

        Args:
            location (str): Location of the Qdrant instance (e.g., ':memory:', 'localhost:6333', or URL).
            collection_name (str): Name of the Qdrant collection.
            embedding_function (Optional[callable]): Custom embedding function. If None (default),
                defaults to FastEmbedEmbeddingFunction with 'BAAI/bge-small-en-v1.5'.
            content_payload_key (str): Key for content payload in Qdrant.
            metadata_payload_key (str): Key for metadata payload in Qdrant.
            collection_options (dict): Options for creating the Qdrant collection.

        Raises:
            ValueError: If required connection parameters are not provided.
        """
        self.location = location
        self.collection_name = collection_name
        self.content_payload_key = content_payload_key
        self.metadata_payload_key = metadata_payload_key
        self.collection_options = collection_options

        # Initialize embedding function
        self.embedding_function = embedding_function or FastEmbedEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.llama_embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        self.embed_dim = len(
            self.embedding_function(["The quick brown fox jumps over the lazy dog."])[0]
        )

        # These will be initialized later
        self.vector_db: QdrantVectorDB | None = None
        self.vector_store: QdrantVectorStore | None = None  # type: ignore[no-any-unimported]
        self.storage_context: StorageContext | None = None  # type: ignore[no-any-unimported]
        self.index: VectorStoreIndex | None = None  # type: ignore[no-any-unimported]

    def _set_up(self, overwrite: bool) -> None:
        """Sets up the Qdrant database, vector store, and storage context.

        This method initializes the vector database using the provided connection details,
        creates a collection (with overwrite if specified), creates a vector store instance,
        and sets the storage context for indexing.

        Args:
            overwrite (bool): Flag indicating whether to overwrite the existing collection.
        """
        logger.info("Setting up the Qdrant database.")
        self.vector_db = VectorDBFactory.create_vector_database(
            db_type="qdrant",
            client_kwargs={"location": self.location},
            embedding_function=self.embedding_function,
            content_payload_key=self.content_payload_key,
            metadata_payload_key=self.metadata_payload_key,
            collection_options=self.collection_options,
        )
        self.vector_db.create_collection(
            collection_name=self.collection_name, overwrite=overwrite
        )
        logger.info("Qdrant vector database created.")

        self.vector_store = QdrantVectorStore(
            client=QdrantClient(location=self.location),
            collection_name=self.collection_name,
            content_payload_key=self.content_payload_key,
            metadata_payload_key=self.metadata_payload_key,
        )
        logger.info("Qdrant vector store created.")
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def _check_existing_collection(self) -> bool:
        """Checks if the specified collection exists in the Qdrant database.

        Returns:
            bool: True if the collection exists; False otherwise.
        """
        try:
            return self.vector_db.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def connect_database(
        self, collection_name: str | None = None, *args: Any, **kwargs: Any
    ) -> bool:
        """Connects to the Qdrant database and initializes the query index from the existing collection.

        This method verifies the existence of the collection, sets up the database connection,
        builds the vector store index, and pings the Qdrant server.

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
                    f"Collection '{self.collection_name}' not found in Qdrant database. "
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
            self.vector_db.client.get_collections()
            logger.info("Connected to Qdrant successfully.")
            return True
        except Exception as error:
            logger.error(f"Failed to connect to Qdrant: {error}")
            return False

    def initialize_collection(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        overwrite: Optional[bool] | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initializes the Qdrant database by creating or overwriting the collection and indexing documents.

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
                    f"Collection '{self.collection_name}' already exists in Qdrant database. "
                    "Set overwrite=True to overwrite it or use connect_database to connect to the existing collection."
                )
            # Set up the database with overwriting if specified
            self._set_up(overwrite=overwrite)
            self.vector_db.client.get_collections()  # Simple ping-like query
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
        """Retrieves the name of the Qdrant collection.

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
