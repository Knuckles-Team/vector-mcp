#!/usr/bin/python
# coding: utf-8

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, List, Dict

if TYPE_CHECKING:
    pass

from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb import ChromaVectorDB
from vector_mcp.vectordb.utils import (
    optional_import_block,
    require_optional_import,
)

from vector_mcp.utils import get_embedding_model
from vector_mcp.vectordb.base import VectorDBFactory

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument

with optional_import_block():
    from chromadb import HttpClient
    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore

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
    """This engine leverages Chromadb to persist document embeddings."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        settings: Optional["Settings"] = None,
        tenant: str | None = None,
        database: str | None = None,
        embedding_function: Any | None = None,
        metadata: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initializes the ChromaDBRetriever."""
        self.host = host
        self.port = port
        self.path = path
        self.settings = settings
        self.tenant = tenant
        self.database = database
        self.metadata = metadata

        self.collection_name = (
            collection_name if collection_name else DEFAULT_COLLECTION_NAME
        )

        self.embed_model = get_embedding_model()

        self.vector_db: ChromaVectorDB | None = None
        self.index: VectorStoreIndex | None = None
        self.vector_store: ChromaVectorStore | None = None
        self.storage_context: StorageContext | None = None

        # Try connection/client creation just to mirror original check logic (optional)
        # But we will rely on VectorDBFactory
        if (not host or not port) and not path:
            # Just logging warning as per original
            logger.warning(
                "Can't connect to remote Chroma client without host or port not. Using an ephemeral, in-memory client (implied by path or lack thereof)."
            )

    def _set_up(self, overwrite: bool) -> None:
        """Set up ChromaDB and LlamaIndex storage."""
        # Using self.client if it was passed? No, constructor just took host/port.
        # We construct client via ChromaVectorDB now.

        # NOTE: ChromaVectorDB takes client object or path or host/port.
        client = None
        if self.host and self.port:
            try:
                client = HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=self.settings,
                    tenant=self.tenant if self.tenant else DEFAULT_TENANT,  # type: ignore
                    database=self.database if self.database else DEFAULT_DATABASE,  # type: ignore
                )
            except Exception:
                pass  # Fallback? Or raise? Original raised.

        self.vector_db = VectorDBFactory.create_vector_database(
            db_type="chroma",
            client=client,
            path=self.path,
            embed_model=self.embed_model,
            collection_name=self.collection_name,
            metadata=self.metadata,
        )

        self.vector_db.create_collection(
            collection_name=self.collection_name, overwrite=overwrite
        )

        self.index = self.vector_db._get_index()

    def initialize_collection(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        document_contents: Sequence[str] | None = None,
        overwrite: Optional[bool] | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initialize the database with the input documents or records."""
        self._set_up(overwrite=overwrite)

        if document_directory or document_paths or document_contents:
            documents = self._load_doc(
                input_dir=document_directory,
                input_docs=document_paths,
                input_contents=document_contents,
            )
            for doc in documents:
                self.index.insert(doc)
        return True

    def connect_database(self, collection_name=None, *args: Any, **kwargs: Any) -> bool:
        """Connect to the database without overwriting the existing collection."""
        if collection_name:
            self.collection_name = collection_name

        self._set_up(overwrite=False)
        return True

    def add_documents(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        document_contents: Sequence[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Sequence["LlamaDocument"]:
        """Add new documents to the underlying database and index."""
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
        self, question: str, number_results: int, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Retrieve information from indexed documents by processing a query."""
        self._validate_query_index()
        similarity_top_k = kwargs.get("number_results", 3)
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

    def get_collection_name(self) -> str:
        """Get the name of the collection used by the query engine."""
        if self.collection_name:
            return self.collection_name
        else:
            raise ValueError("Collection name not set.")

    def _validate_query_index(self) -> None:
        """Ensures an index exists."""
        if not hasattr(self, "index") or self.index is None:
            raise Exception(
                "Query index is not initialized. Please call initialize_collection or connect_database first."
            )

    def bm25_query(
        self, question: str, number_results: int, *args: Any, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        self._validate_query_index()
        # ChromaVectorDB lexical_search returns list of list of (Document, score)
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

    def _load_doc(
        self,
        input_dir: Path | str | None = None,
        input_docs: Sequence[Path | str] | None = None,
        input_contents: Sequence[str] | None = None,
    ) -> Sequence["LlamaDocument"]:
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


if TYPE_CHECKING:
    from .retriever import RAGQueryEngine

    def _check_implement_protocol(o: ChromaDBRetriever) -> RAGQueryEngine:
        return o
