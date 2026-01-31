#!/usr/bin/python
# coding: utf-8

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llama_index.core.schema import Document as LlamaDocument

__all__ = ["RAGRetriever"]


@runtime_checkable
class RAGRetriever(Protocol):
    """A protocol class that represents a document ingestion and query engine on top of an underlying database.

    This interface defines the basic methods for RAG.
    """

    def initialize_collection(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        document_contents: Sequence[str] | None = None,
        overwrite: Optional[bool] | None = True,
        collection_name: Optional[str] | None = "memory",
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initialize the database with the input documents or records.

        This method initializes database with the input documents or records.
        Usually, it takes the following steps:\n
        1. connecting to a database.\n
        2. insert records.\n
        3. build indexes etc.\n

        Args:\n
            document_directory (Optional[Union[Path, str]]): A directory containing documents to be ingested.\n
            document_paths (Optional[Sequence[Union[Path, str]]]): A list of paths or URLs to documents to be ingested.\n
            document_contents (Optional[Sequence[str]]): A list of string contents to be ingested directly.\n
            *args: Any additional arguments\n
            **kwargs: Any additional keyword arguments\n
        Returns:\n
            bool: True if initialization is successful, False otherwise\n
        """
        ...

    def add_documents(
        self,
        document_directory: Path | str | None = None,
        document_paths: Sequence[Path | str] | None = None,
        document_contents: Sequence[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Sequence["LlamaDocument"]:
        """Add new documents to the underlying data store.
        Returns:
            List: List of documents ingested"""
        ...

    def connect_database(self, *args: Any, **kwargs: Any) -> bool:
        """Connect to the database.

        Args:
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments
        Returns:
            bool: True if connection is successful, False otherwise
        """
        ...

    def query(
        self, question: str, number_results: int, *args: Any, **kwargs: Any
    ) -> str:
        """Transform a string format question into database query and return the result.

        Args:
            question: a string format question
            number_results: number of results to return
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments
        """
        ...

    def bm25_query(
        self, question: str, number_results: int, *args: Any, **kwargs: Any
    ) -> str:
        """Transform a string format question into database query and return the result using BM25.

        Args:
            question: a string format question
            number_results: number of results to return
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments
        """
        ...
