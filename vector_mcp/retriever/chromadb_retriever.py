#!/usr/bin/python
# coding: utf-8

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb import ChromaVectorDB
from vector_mcp.vectordb.utils import optional_import_block, require_optional_import
from vector_mcp.vectordb.base import VectorDBFactory

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

with optional_import_block():
    from chromadb import HttpClient
    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

__all__ = ["ChromaDBRetriever"]

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


@require_optional_import(["chromadb", "llama_index"], "rag")
class ChromaDBRetriever(RAGRetriever):
    """This engine leverages Chromadb to persist document embeddings in a named collection
    and LlamaIndex's VectorStoreIndex to efficiently index and retrieve documents, and generate an answer in response
    to natural language queries. Collection can be regarded as an abstraction of group of documents in the database.

    It expects a Chromadb server to be running and accessible at the specified host and port.
    Refer to this [link](https://docs.trychroma.com/production/containers/docker) for running Chromadb in a Docker container.
    If the host and port are not provided, the engine will create an in-memory ChromaDB client.


    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        settings: Optional["Settings"] = None,
        tenant: str | None = None,
        database: str | None = None,
        embedding_function: "EmbeddingFunction[Any] | None" = None,  # type: ignore[type-arg]
        metadata: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initializes the ChromaDBRetriever with db_path, metadata, and embedding function and llm.

        Args:
            host: The host address of the ChromaDB server. Default is localhost.
            port: The port number of the ChromaDB server. Default is 8000.
            settings: A dictionary of settings to communicate with the chroma server. Default is None.
            tenant: The tenant to use for this client. Defaults to the default tenant.
            database: The database to use for this client. Defaults to the default database.
            embedding_function: A callable that converts text into vector embeddings. Default embedding uses Sentence Transformers model all-MiniLM-L6-v2.
                For more embeddings that ChromaDB support, please refer to [embeddings](https://docs.trychroma.com/docs/embeddings/embedding-functions)
            metadata: A dictionary containing configuration parameters for the Chromadb collection.
                This metadata is typically used to configure the HNSW indexing algorithm. Defaults to `{"hnsw:space": "ip", "hnsw:construction_ef": 30, "hnsw:M": 32}`
                For more details about the default metadata, please refer to [HNSW configuration](https://cookbook.chromadb.dev/core/configuration/#hnsw-configuration)
            collection_name (str): The unique name for the Chromadb collection. If omitted, a constant name will be used. Populate this to reuse previous ingested data.
        """
        self.retriever = None
        self.index = None
        if not host or not port:
            logger.warning(
                "Can't connect to remote Chroma client without host or port not. Using an ephemeral, in-memory client."
            )
            self.client = None
        else:
            try:
                self.client = HttpClient(
                    host=host,
                    port=port,
                    settings=settings,
                    tenant=tenant if tenant else DEFAULT_TENANT,  # type: ignore[arg-type, no-any-unimported]
                    database=database if database else DEFAULT_DATABASE,  # type: ignore[arg-type, no-any-unimported]
                )
            except Exception as e:
                raise ValueError(f"Failed to connect to the ChromaDB client: {e}")
        self.embedding_function = (
            embedding_function
            if embedding_function
            else SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        )
        self.llama_embedding = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db_config = {
            "client": self.client,
            "embedding_function": embedding_function,
            "metadata": metadata,
            "path": path,
        }
        self.collection_name = (
            collection_name if collection_name else DEFAULT_COLLECTION_NAME
        )
        self.vector_db: ChromaVectorDB | None = None
        self.vector_store: ChromaVectorStore | None = None  # type: ignore[no-any-unimported]
        self.storage_context: StorageContext | None = None  # type: ignore[no-any-unimported]
        self.index: VectorStoreIndex | None = None  # type: ignore[no-any-unimported]

    def initialize_collection(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        overwrite: Optional[bool] | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initialize the database with the input documents or records.

        Args:
            document_directory: A dir of input documents to create records in the database.
            document_paths: A sequence of input documents to create records in the database.
            overwrite: If True, overwrite the existing collection.

        Returns:
            bool: True if initialization is successful
        """
        self._set_up(overwrite=overwrite)
        documents = self._load_doc(
            input_dir=document_directory, input_docs=document_paths
        )
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=self.storage_context,
            embed_model=self.llama_embedding,
        )
        return True

    def connect_database(self, collection_name=None, *args: Any, **kwargs: Any) -> bool:
        """Connect to the database without overwriting the existing collection.

        Returns:
            bool: True if connection is successful
        """
        if collection_name:
            self.collection_name = collection_name

        self._set_up(overwrite=False)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
            embed_model=self.llama_embedding,  # Use local embedding model
        )
        return True

    def add_documents(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Sequence["LlamaDocument"]:
        """Add new documents to the underlying database and index.

        Args:
            document_directory: A dir of input documents to add.
            document_paths: A sequence of input documents to add.
        Returns:
            List: List of documents
        """
        self._validate_query_index()
        documents = self._load_doc(
            input_dir=document_directory, input_docs=document_paths
        )
        for doc in documents:
            self.index.insert(
                doc, embed_model=self.llama_embedding
            )  # Use local embedding model
        return documents

    def query(self, question: str, number_results: int, **kwargs: Any) -> str:
        """Retrieve information from indexed documents by processing a query.

        Args:
            question: A natural language query string.
            number_results: Number of results to return as an integer

        Returns:
            A string containing the response.
        """
        self._validate_query_index()
        similarity_top_k = kwargs.get("number_results", 3)
        self.retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        response = self.retriever.retrieve(str_or_query_bundle=question)

        if str(response) == EMPTY_RESPONSE_TEXT:
            return EMPTY_RESPONSE_REPLY

        return str(response)

    def get_collection_name(self) -> str:
        """Get the name of the collection used by the query engine.

        Returns:
            The name of the collection.
        """
        if self.collection_name:
            return self.collection_name
        else:
            raise ValueError("Collection name not set.")

    def _validate_query_index(self) -> None:
        """Ensures an index exists."""
        if not hasattr(self, "index"):
            raise Exception(
                "Query index is not initialized. Please call initialize_collection or connect_database first."
            )

    def _set_up(self, overwrite: bool) -> None:
        """Set up ChromaDB and LlamaIndex storage.

        Args:
            overwrite: If True, overwrite the existing collection.
        """
        self.vector_db = VectorDBFactory().create_vector_database(
            db_type="chroma", **self.db_config
        )
        self.collection = self.vector_db.create_collection(
            collection_name=self.collection_name, overwrite=overwrite
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def _load_doc(
        self, input_dir: Path | str | None, input_docs: Sequence[Path | str] | None
    ) -> Sequence["LlamaDocument"]:
        """Load documents from a directory and/or a sequence of file paths.

        Args:
            input_dir: The directory containing documents to be loaded.
            input_docs: A sequence of individual file paths to load.

        Returns:
            A sequence of documents loaded as LlamaDocument objects.
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


if TYPE_CHECKING:
    from .retriever import RAGQueryEngine

    def _check_implement_protocol(o: ChromaDBRetriever) -> RAGQueryEngine:
        return o
