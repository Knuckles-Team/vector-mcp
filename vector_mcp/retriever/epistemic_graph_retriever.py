#!/usr/bin/python
"""RAG retriever over the native epistemic-graph vector backend.

Unlike the ChromaDB/Postgres retrievers (which drive a LlamaIndex
``VectorStoreIndex``), this retriever talks to :class:`EpistemicGraphVectorDB`
**directly** — the engine owns the ANN index, so documents are embedded + inserted
and queried through the backend's ``insert_documents`` / ``semantic_search`` /
``lexical_search`` rather than a client-side index.
"""

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_utilities import create_embedding_model

from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb.base import Document, VectorDB, VectorDBFactory
from vector_mcp.vectordb.db_utils import optional_import_block, require_optional_import

with optional_import_block():
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.schema import Document as LlamaDocument

__all__ = ["EpistemicGraphRetriever"]

DEFAULT_COLLECTION_NAME = "memory"

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(["epistemic_graph", "agent_utilities"], "epistemic-graph")
class EpistemicGraphRetriever(RAGRetriever):
    """A RAG retriever backed by the native epistemic-graph engine."""

    def __init__(
        self,
        socket_path: str | None = None,
        auth_secret: str | None = None,
        tcp_addr: str | None = None,
        _embedding_function: Any | None = None,
        metadata: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initialize the EpistemicGraphRetriever."""
        self.socket_path = socket_path
        self.auth_secret = auth_secret
        self.tcp_addr = tcp_addr
        self.metadata = metadata
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.embed_model = create_embedding_model()
        self.vector_db: VectorDB | None = None
        # No client-side index — the engine owns the ANN index.
        self.index = None
        self.active_collection = self.collection_name

    def _set_up(self, overwrite: bool) -> None:
        """Create the epistemic-graph backend and (re)initialize the collection."""
        self.vector_db = VectorDBFactory.create_vector_database(
            db_type="epistemic_graph",
            embed_model=self.embed_model,
            collection_name=self.collection_name,
            socket_path=self.socket_path,
            auth_secret=self.auth_secret,
            tcp_addr=self.tcp_addr,
            metadata=self.metadata,
        )
        self.vector_db.create_collection(
            collection_name=self.collection_name, overwrite=overwrite
        )
        self.active_collection = self.collection_name

    def _validate_query_index(self) -> None:
        if self.vector_db is None:
            raise Exception(
                "Vector database is not initialized. Call initialize_collection or "
                "connect_database first."
            )

    def _to_documents(self, llama_docs: Sequence["LlamaDocument"]) -> list[Document]:
        docs: list[Document] = []
        for i, doc in enumerate(llama_docs):
            doc_id = getattr(doc, "doc_id", None) or getattr(doc, "id_", None) or str(i)
            docs.append(
                Document(
                    id=str(doc_id),
                    content=doc.get_content(),
                    metadata=dict(getattr(doc, "metadata", {}) or {}),
                )
            )
        return docs

    def initialize_collection(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        document_contents: Sequence[str] | None = None,
        overwrite: bool | None = True,
        collection_name: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initialize the collection and optionally ingest documents."""
        if collection_name:
            self.collection_name = collection_name
        self._set_up(overwrite=True if overwrite is None else overwrite)

        if document_directory or document_paths or document_contents:
            llama_docs = self._load_doc(
                input_dir=document_directory,
                input_docs=document_paths,
                input_contents=document_contents,
            )
            assert self.vector_db is not None
            self.vector_db.insert_documents(
                self._to_documents(llama_docs), collection_name=self.collection_name
            )
        return True

    def connect_database(self, collection_name=None, *args: Any, **kwargs: Any) -> bool:
        """Connect to an existing collection without overwriting it."""
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
        """Embed and insert new documents into the engine."""
        self._validate_query_index()
        llama_docs = self._load_doc(
            input_dir=document_directory,
            input_docs=document_paths,
            input_contents=document_contents,
        )
        assert self.vector_db is not None
        self.vector_db.insert_documents(
            self._to_documents(llama_docs), collection_name=self.collection_name
        )
        return llama_docs

    def query(
        self, question: str, number_results: int, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Semantic (vector) search over the engine's native ANN index."""
        self._validate_query_index()
        assert self.vector_db is not None
        n = kwargs.get("number_results", number_results)
        results = self.vector_db.semantic_search(
            queries=[question], collection_name=self.collection_name, n_results=n
        )
        return self._format(results[0] if results else [])

    def bm25_query(
        self, question: str, number_results: int, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Lexical search over the engine's term index."""
        self._validate_query_index()
        assert self.vector_db is not None
        results = self.vector_db.lexical_search(
            queries=[question],
            collection_name=self.collection_name,
            n_results=number_results,
        )
        return self._format(results[0] if results else [])

    @staticmethod
    def _format(doc_scores) -> list[dict[str, Any]]:
        out = []
        for doc, score in doc_scores:
            out.append(
                {
                    "text": doc.get("content", ""),
                    "score": score,
                    "id": doc.get("id", ""),
                    "metadata": doc.get("metadata"),
                }
            )
        return out

    def get_collection_name(self) -> str:
        if self.collection_name:
            return self.collection_name
        raise ValueError("Collection name not set.")

    def _load_doc(
        self,
        input_dir: Path | str | None = None,
        input_docs: Sequence[Path | str] | None = None,
        input_contents: Sequence[str] | None = None,
    ) -> Sequence["LlamaDocument"]:
        loaded_documents = []
        if input_dir:
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory not found: {input_dir}")
            loaded_documents.extend(
                SimpleDirectoryReader(input_dir=input_dir).load_data()
            )
        if input_docs:
            for doc in input_docs:
                if not os.path.exists(doc):
                    raise ValueError(f"Document file not found: {doc}")
            loaded_documents.extend(
                SimpleDirectoryReader(input_files=input_docs).load_data()
            )
        if input_contents:
            for content in input_contents:
                loaded_documents.append(LlamaDocument(text=content))
        if not input_dir and not input_docs and not input_contents:
            raise ValueError(
                "No input directory, docs, or content provided! You must provide at "
                "least one source."
            )
        return loaded_documents


if TYPE_CHECKING:
    from .retriever import RAGRetriever as RAGQueryEngine

    def _check_implement_protocol(o: EpistemicGraphRetriever) -> RAGQueryEngine:
        return o
