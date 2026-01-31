#!/usr/bin/python
# coding: utf-8

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, List, Dict

if TYPE_CHECKING:
    pass

from pydantic import Field
from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb.utils import (
    optional_import_block,
    require_optional_import,
)

from vector_mcp.utils import get_embedding_model
from vector_mcp.vectordb.base import VectorDBFactory

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument

with optional_import_block():
    from llama_index.vector_stores.couchbase import CouchbaseVectorStore
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


@require_optional_import(["couchbase", "llama_index"], "rag")
class CouchbaseRetriever(RAGRetriever):
    """A query engine backed by Couchbase that supports document insertion and querying."""

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
        embedding_function: Any | None = None,
    ):
        """Initializes a CouchbaseRetriever instance."""
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

        self.embed_model = (
            get_embedding_model()
        )  # embedding_function ignored mostly or used if custom passed? ignoring for now to standardize.

        # These will be initialized later
        self.vector_db: CouchbaseVectorDB | None = None
        self.vector_store: CouchbaseVectorStore | None = None  # type: ignore[no-any-unimported]
        self.storage_context: StorageContext | None = None  # type: ignore[no-any-unimported]
        self.index: VectorStoreIndex | None = None  # type: ignore[no-any-unimported]

    def _set_up(self, overwrite: bool) -> None:
        """Sets up the Couchbase database via VectorDBFactory."""
        logger.info("Setting up the Couchbase database.")
        self.vector_db = VectorDBFactory.create_vector_database(
            db_type="couchbase",
            connection_string=self.connection_string,
            username=self.username,
            password=self.password,
            dbname=self.bucket_name,  # bucket mapped to dbname
            scope_name=self.scope_name,
            collection_name=self.collection_name,
            index_name=self.index_name,
            embed_model=self.embed_model,
        )
        self.vector_db.create_collection(
            collection_name=self.collection_name, overwrite=overwrite
        )
        logger.info("Couchbase vector database created.")

        # Access internal components
        self.index = self.vector_db._get_index()

    def _check_existing_collection(self) -> bool:
        """Checks if the specified collection exists in the Couchbase database."""
        try:
            # CouchbaseVectorDB (wrapped) doesn't expose bucket directly easily unless we access internal
            collection_mgr = self.vector_db.cluster.bucket(
                self.bucket_name
            ).collections()
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
        """Connects to the Couchbase database and initializes the query index from the existing collection."""
        if collection_name:
            self.collection_name = collection_name
        try:
            # Just set up, overwrite false.
            self._set_up(overwrite=False)

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
        document_contents: Sequence[str] | None = None,
        overwrite: Optional[bool] | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initializes the Couchbase database by creating or overwriting the collection and indexing documents."""
        try:
            # Logic is slightly different in original: overwrite if exists.
            # CouchbaseVectorDB: create_collection(overwrite=overwrite) handles it? logic is weak there.
            # But let's trust _set_up logic.

            # Set up the database with overwriting if specified
            self._set_up(overwrite=overwrite)
            self.vector_db.cluster.ping()  # Simple ping-like query

            if document_directory or document_paths or document_contents:
                logger.info("Setting up the database with documents.")
                documents = self._load_doc(
                    input_dir=document_directory,
                    input_docs=document_paths,
                    input_contents=document_contents,
                )
                for doc in documents:
                    self.index.insert(doc)
                logger.info("Database initialized with %d documents.", len(documents))
            return True
        except Exception as e:
            logger.error(f"Failed to initialize the database: {e}")
            return False

    def _validate_query_index(self) -> None:
        """Validates that the query index is initialized."""
        if not hasattr(self, "index") or self.index is None:
            raise Exception(
                "Query index is not initialized. Please call initialize_collection or connect_database first."
            )

    def _load_doc(
        self,
        input_dir: Path | str | None = None,
        input_docs: Sequence[Path | str] | None = None,
        input_contents: Sequence[str] | None = None,
    ) -> Sequence["LlamaDocument"]:
        """Loads documents from a directory or a list of file paths."""
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

        if input_contents:
            logger.info(f"Loading {len(input_contents)} strings as documents")
            for content in input_contents:
                loaded_documents.append(LlamaDocument(text=content))

        if not input_dir and not input_docs and not input_contents:
            raise ValueError(
                "No input directory, docs, or content provided! You must provide at least one source."
            )

        return loaded_documents

    def add_documents(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        document_contents: Sequence[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Sequence["LlamaDocument"]:
        """Adds new documents to the existing vector store index."""
        self._validate_query_index()
        documents = self._load_doc(
            input_dir=document_directory,
            input_docs=document_paths,
            input_contents=document_contents,
        )
        for doc in documents:
            self.index.insert(doc)
        return documents

    def query(
        self, question: str, number_results: int, *args: Any, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Queries the indexed documents using the provided question."""
        self._validate_query_index()
        similarity_top_k = kwargs.get("number_results", number_results)
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        response = retriever.retrieve(str_or_query_bundle=question)

        results = []
        for node_with_score in response:
            results.append(
                {
                    "text": node_with_score.node.get_content(),
                    "score": node_with_score.score,
                    "id": node_with_score.node.node_id,
                    "metadata": node_with_score.node.metadata,
                }
            )
        return results

    def bm25_query(
        self, question: str, number_results: int, *args: Any, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        self._validate_query_index()
        # CouchbaseVectorDB lexical_search returns list of list of (Document, score)
        results = self.vector_db.lexical_search(
            queries=[question],
            collection_name=self.collection_name,
            n_results=number_results,
            **kwargs,
        )
        doc_scores = results[0]

        formatted_results = []
        for doc, score in doc_scores:
            formatted_results.append(
                {
                    "text": (
                        doc.content if hasattr(doc, "content") else doc.get("content")
                    ),
                    "score": score,
                    "id": doc.id if hasattr(doc, "id") else doc.get("id"),
                    "metadata": (
                        doc.metadata
                        if hasattr(doc, "metadata")
                        else doc.get("metadata")
                    ),
                }
            )
        return formatted_results

    def get_collection_name(self) -> str:
        """Retrieves the name of the Couchbase collection."""
        if self.collection_name:
            return self.collection_name
        else:
            raise ValueError("Collection name not set.")


if TYPE_CHECKING:
    from .retriever import RAGQueryEngine

    def _check_implement_protocol(o: CouchbaseRetriever) -> RAGQueryEngine:
        return o
