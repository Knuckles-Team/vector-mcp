#!/usr/bin/python
# coding: utf-8

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, List, Dict

if TYPE_CHECKING:
    pass

from vector_mcp.retriever.retriever import RAGRetriever

from vector_mcp.vectordb.utils import (
    optional_import_block,
    require_optional_import,
)

from vector_mcp.utils import get_embedding_model
from vector_mcp.vectordb.base import VectorDBFactory

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument

with optional_import_block():
    from vector_mcp.vectordb.pgvector import PGVectorDB

__all__ = ["PGVectorRetriever"]

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


@require_optional_import(["pgvector", "psycopg", "llama_index"], "rag")
class PGVectorRetriever(RAGRetriever):
    """A query engine backed by PGVector that supports document insertion and querying."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        connection_string: Optional[str] = None,
        host: Optional[Union[str, int]] = None,
        port: Optional[Union[str, int]] = None,
        dbname: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database_name: str | None = None,
        embedding_function: Any | None = None,
        collection_name: str | None = None,
    ):
        """Initializes a PGVectorRetriever instance."""

        # Connection logic is now mostly handled by PGVectorDB, but we pass params.
        self.connection_string = connection_string
        self.host = str(host) if host else None
        self.port = str(port) if port else None
        self.dbname = dbname
        self.username = username
        self.password = password

        self.database_name = database_name or dbname
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME

        self.embed_model = get_embedding_model()

        self.vector_db: PGVectorDB | None = None
        self.index: VectorStoreIndex | None = None

    def _set_up(self, overwrite: bool) -> None:
        """Sets up the PGVector database via PGVectorDB."""
        logger.info("Setting up the database.")
        self.vector_db: PGVectorDB = VectorDBFactory.create_vector_database(
            db_type="pgvector",
            connection_string=self.connection_string,
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            username=self.username,
            password=self.password,
            embed_model=self.embed_model,
            collection_name=self.collection_name,
        )
        self.vector_db.create_collection(
            collection_name=self.collection_name, overwrite=overwrite
        )
        logger.info("Vector database created.")

        # PGVectorDB initializes vector_store and storage_context
        # We can access internal index if available or create one.
        # PGVectorDB lazy loads index, so we can trigger it.
        self.index = self.vector_db._get_index()

    def connect_database(
        self,
        collection_name=None,
        ensure_exists: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        if collection_name:
            self.collection_name = collection_name
        try:
            self._set_up(overwrite=False)
            logger.info("Connected to PostgreSQL successfully.")
            return True
        except Exception as error:
            logger.error("Failed to connect to PostgreSQL: %s", error)
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
        try:
            # Logic is slightly different in original: overwrite if exists.
            # PGVectorDB: create_collection(overwrite=overwrite) handles it.
            self._set_up(overwrite=overwrite)

            if document_directory or document_paths or document_contents:
                logger.info("Setting up the database with documents.")
                documents = self._load_doc(
                    input_dir=document_directory,
                    input_docs=document_paths,
                    input_contents=document_contents,
                )
                # Use vector_db insert to handle index update
                # Using internal index insert
                for doc in documents:
                    self.index.insert(doc)
                # Or just insert_documents?
                # self.vector_db.insert_documents(...) expects dicts, we have LlamaDocuments.
                # Better to use index directly as before.

                logger.info("Database initialized with %d documents.", len(documents))
            return True
        except Exception as e:
            logger.error("Failed to initialize the database: %s", e)
            return False

    def _validate_query_index(self) -> None:
        if not hasattr(self, "index") or self.index is None:
            raise Exception(
                "Query index is not initialized. Please call initialize_collection or connect_database first."
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
                SimpleDirectoryReader(input_files=input_docs).load_data()
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
    ) -> Sequence["LlamaDocument"]:
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
        # PGVectorDB lexical_search returns list of list of (Document, score)
        results = self.vector_db.lexical_search(
            queries=[question],
            collection_name=self.collection_name,
            n_results=number_results,
            **kwargs,
        )
        # results[0] because we sent 1 question
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
        if self.collection_name:
            return self.collection_name
        else:
            raise ValueError("Collection name not set.")


if TYPE_CHECKING:
    from .retriever import RAGQueryEngine

    def _check_implement_protocol(o: PGVectorRetriever) -> RAGQueryEngine:
        return o
