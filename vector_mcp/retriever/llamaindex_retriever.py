#!/usr/bin/python


import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_utilities import create_embedding_model

from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.vectordb.db_utils import (
    optional_import_block,
    require_optional_import,
)

with optional_import_block():
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.core.vector_stores.types import BasePydanticVectorStore

__all__ = ["LlamaIndexRetriever"]

EMPTY_RESPONSE_TEXT = "Empty Response"
EMPTY_RESPONSE_REPLY = (
    "Sorry, I couldn't find any information on that. "
    "If you haven't ingested any documents, please try that."
)


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import("llama_index", "rag")
class LlamaIndexRetriever(RAGRetriever):
    """This engine leverages LlamaIndex's VectorStoreIndex to efficiently index and retrieve documents, and generate an answer in response
    to natural language queries. It use any LlamaIndex [vector store](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/).

    By default the engine will use OpenAI's GPT-4o model (use the `llm` parameter to change that).
    """

    def __init__(
        self,
        vector_store: "BasePydanticVectorStore",
        file_reader_class: type["SimpleDirectoryReader"] | None = None,
    ) -> None:
        """Initializes the LlamaIndexRetriever with the given vector store.

        Args:
            vector_store: The vector store to use for indexing and querying documents.
            llm: LLM model used by LlamaIndex for query processing. You can find more supported LLMs at [LLM](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/).
            file_reader_class: The file reader class to use for loading documents. Only SimpleDirectoryReader is currently supported.
        """
        self.vector_store = vector_store
        self.file_reader_class = (
            file_reader_class if file_reader_class else SimpleDirectoryReader
        )

        self.embed_model = create_embedding_model()

    def initialize_collection(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        document_contents: Sequence[str] | None = None,
        overwrite: bool | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initialize the database with the input documents or records."""
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        documents = self._load_doc(
            input_dir=document_directory,
            input_docs=document_paths,
            input_contents=document_contents,
        )
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
        )
        return True

    def connect_database(self, *args: Any, **kwargs: Any) -> bool:
        """Connect to the database."""
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
        )

        return True

    def add_documents(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        document_contents: Sequence[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Sequence["LlamaDocument"]:
        """Add new documents to the underlying database and add to the index."""
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
    ) -> list[dict[str, Any]]:
        """Retrieve information from indexed documents by processing a query using the engine's LLM."""
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
    ) -> list[dict[str, Any]]:
        # LlamaIndexRetriever doesn't seem to have a specific BM25 implementation here,
        # fallback to regular query for now or raise NotImplementedError.
        return self.query(question, number_results, *args, **kwargs)

    def _validate_query_index(self) -> None:
        """Ensures an index exists"""
        if not hasattr(self, "index"):
            raise Exception(
                "Query index is not initialized. Please call init_db or connect_database first."
            )

    def _load_doc(
        self,
        input_dir: Path | str | None = None,
        input_docs: Sequence[Path | str] | None = None,
        input_contents: Sequence[str] | None = None,
    ) -> Sequence["LlamaDocument"]:
        """Load documents from a directory and/or a sequence of file paths."""
        loaded_documents: list[LlamaDocument] = []
        if input_dir:
            logger.info(f"Loading docs from directory: {input_dir}")
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory not found: {input_dir}")
            loaded_documents.extend(
                self.file_reader_class(input_dir=input_dir).load_data()
            )

        if input_docs:
            for doc in input_docs:
                logger.info(f"Loading input doc: {doc}")
                if not os.path.exists(doc):
                    raise ValueError(f"Document file not found: {doc}")
            loaded_documents.extend(
                self.file_reader_class(input_files=input_docs).load_data()
            )

        if input_contents:
            logger.info(f"Loading {len(input_contents)} strings as documents")
            for content in input_contents:
                loaded_documents.append(LlamaDocument(text=content))

        if not input_dir and not input_docs and not input_contents:
            raise ValueError("No input directory, docs, or content provided!")

        return loaded_documents


if TYPE_CHECKING:
    from .retriever import RAGRetriever as RAGQueryEngine

    def _check_implement_protocol(o: LlamaIndexRetriever) -> RAGQueryEngine:
        return o
