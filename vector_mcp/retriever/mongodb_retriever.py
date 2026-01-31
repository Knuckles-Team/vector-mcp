#!/usr/bin/python
# coding: utf-8

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, List, Dict

if TYPE_CHECKING:
    pass

from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb.base import VectorDBFactory
from vector_mcp.vectordb.mongodb import MongoDBAtlasVectorDB
from vector_mcp.vectordb.utils import (
    optional_import_block,
    require_optional_import,
)

from vector_mcp.utils import get_embedding_model

with optional_import_block():
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
    from pymongo import MongoClient

__all__ = ["MongoDBRetriever"]

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


@require_optional_import(["pymongo", "llama_index"], "rag")
class MongoDBRetriever(RAGRetriever):
    """A query engine backed by MongoDB Atlas that supports document insertion and querying."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        connection_string: str,
        database_name: str | None = None,
        embedding_function: Any | None = None,
        embedding_model: Union["BaseEmbedding", str] | None = None,
        collection_name: str | None = None,
    ):
        """Initializes a MongoDBRetriever instance."""
        if not connection_string:
            raise ValueError("Connection string is required to connect to MongoDB.")

        self.connection_string = connection_string
        # ToDo: Is it okay if database_name is None?
        self.database_name = database_name
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME

        self.embed_model = (
            get_embedding_model()
        )  # embedding_function ignored mostly or used if custom passed? ignoring for now to standardize.

        # These will be initialized later.
        self.vector_db: MongoDBAtlasVectorDB | None = None
        self.semantic_search_engine: MongoDBAtlasVectorSearch | None = None  # type: ignore[no-any-unimported]
        self.storage_context: StorageContext | None = None  # type: ignore[no-any-unimported]
        self.index: VectorStoreIndex | None = None  # type: ignore[no-any-unimported]

    def _set_up(self, overwrite: bool) -> None:
        """Sets up the MongoDB vector database via VectorDBFactory."""
        logger.info("Setting up the database.")
        self.vector_db: MongoDBAtlasVectorDB = VectorDBFactory.create_vector_database(  # type: ignore[assignment, no-redef]
            db_type="mongodb",
            connection_string=self.connection_string,
            dbname=self.database_name,  # Note: using dbname as per factory arg usually
            embed_model=self.embed_model,
            collection_name=self.collection_name,
        )
        self.vector_db.create_collection(
            collection_name=self.collection_name, overwrite=overwrite
        )
        logger.info("Vector database created.")

        # Access internal LlamaIndex components from VectorDB wrapper
        self.index = self.vector_db._get_index()

    def _check_existing_collection(self) -> bool:
        """Checks if the specified collection exists in the MongoDB database."""
        client: MongoClient[Any] = MongoClient(self.connection_string)  # type: ignore[no-any-unimported]
        db = client[self.database_name or "default_db"]  # type: ignore[index]
        return self.collection_name in db.list_collection_names()

    def connect_database(self, *args: Any, **kwargs: Any) -> bool:
        """Connects to the MongoDB database and initializes the query index from the existing collection."""
        try:
            # Check if the target collection exists.
            if not self._check_existing_collection():
                raise ValueError(
                    f"Collection '{self.collection_name}' not found in database '{self.database_name}'. "
                    "Please run init_db to create a new collection."
                )
            # Reinitialize without overwriting the existing collection.
            self._set_up(overwrite=False)

            self.vector_db.mongo_client.admin.command("ping")  # type: ignore[union-attr]
            logger.info("Connected to MongoDB successfully.")
            return True
        except Exception as error:
            logger.error("Failed to connect to MongoDB: %s", error)
            return False

    def initialize_collection(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        document_contents: Sequence[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initializes the MongoDB database by creating or overwriting the collection and indexing documents."""
        try:
            # Check if the collection already exists.
            if self._check_existing_collection():
                logger.warning(
                    f"Collection '{self.collection_name}' already exists in database '{self.database_name}'. "
                    "Please use connect_database to connect to the existing collection or use init_db to overwrite it."
                )
            # Set up the database with overwriting.
            self._set_up(overwrite=True)
            self.vector_db.mongo_client.admin.command("ping")  # type: ignore[union-attr]
            # Gather document paths.
            logger.info("Setting up the database with existing collection.")

            if document_directory or document_paths or document_contents:
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
            logger.error("Failed to initialize the database: %s", e)
            return False

    def _validate_query_index(self) -> None:
        """Validates that the query index is initialized."""
        if not hasattr(self, "index") or self.index is None:
            raise Exception(
                "Query index is not initialized. Please call init_db or connect_database first."
            )

    def _load_doc(  # type: ignore[no-any-unimported]
        self,
        input_dir: Path | str | None = None,
        input_docs: Sequence[Path | str] | None = None,
        input_contents: Sequence[str] | None = None,
    ) -> Sequence["LlamaDocument"]:
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
            loaded_documents.extend(
                SimpleDirectoryReader(input_files=input_docs).load_data()  # type: ignore[arg-type]
            )

        if input_contents:
            logger.info("Loading %d strings as documents", len(input_contents))
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
    ) -> None:
        """Adds new documents to the existing vector store index."""
        self._validate_query_index()
        documents = self._load_doc(
            input_dir=document_directory,
            input_docs=document_paths,
            input_contents=document_contents,
        )
        for doc in documents:
            self.index.insert(doc)  # type: ignore[union-attr]

    def query(self, question: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Queries the indexed documents using the provided question."""
        self._validate_query_index()
        similarity_top_k = kwargs.get("n_results", 10)
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
        # MongoDBAtlasVectorDB lexical_search returns list of list of (Document, score)
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
        """Retrieves the name of the MongoDB collection."""
        if self.collection_name:
            return self.collection_name
        else:
            raise ValueError("Collection name not set.")


if TYPE_CHECKING:
    from .retriever import RAGQueryEngine

    def _check_implement_protocol(o: MongoDBRetriever) -> RAGQueryEngine:
        return o
