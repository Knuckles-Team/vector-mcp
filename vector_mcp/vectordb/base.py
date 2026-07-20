#!/usr/bin/python

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict, runtime_checkable

Metadata = Mapping[str, Any] | None
Vector = Sequence[float] | Sequence[int]
ItemID = str | int


class Document(TypedDict, total=False):
    """A Document is a record in the vector database.

    id: ItemID | the unique identifier of the document.
    content: str | the text content of the chunk.
    metadata: Metadata, Optional | contains approved non-identifying document attributes.
    embedding: Vector, Optional | the vector representation of the content.
    """

    id: ItemID
    content: str
    metadata: Metadata | None
    embedding: Vector | None


"""QueryResults is the response from the vector database for a query/queries.
A query is a list containing one string while queries is a list containing multiple strings.
The response is a list of query results, each query result is a list of tuples containing the document and the distance.
"""
QueryResults = list[list[tuple[Document, float]]]


def document_embeddings(docs: list[Document], embed_model: Any) -> list[list[float]]:
    """Resolve one finite, dimension-consistent vector per document.

    Missing vectors use the model's batch API when available, collapsing up to the
    public ingestion bound into one model request instead of one request per document.
    """

    resolved: list[list[float] | None] = [None] * len(docs)
    missing: list[int] = []
    texts: list[str] = []
    for index, document in enumerate(docs):
        supplied = document.get("embedding")
        if supplied is None:
            missing.append(index)
            texts.append(str(document["content"]))
            continue
        resolved[index] = [float(value) for value in supplied]

    if missing:
        batch = getattr(embed_model, "get_text_embedding_batch", None)
        generated = (
            batch(texts)
            if callable(batch)
            else [embed_model.get_text_embedding(text) for text in texts]
        )
        if len(generated) != len(missing):
            raise ValueError("embedding_batch_invalid")
        for index, vector in zip(missing, generated, strict=True):
            resolved[index] = [float(value) for value in vector]

    vectors = [vector for vector in resolved if vector is not None]
    if len(vectors) != len(docs):
        raise ValueError("embedding_batch_invalid")
    dimensions = {len(vector) for vector in vectors}
    if (
        not vectors
        or dimensions == {0}
        or len(dimensions) != 1
        or any(not math.isfinite(value) for vector in vectors for value in vector)
    ):
        raise ValueError("embedding_invalid")
    return vectors


@runtime_checkable
class VectorDB(Protocol):
    """Abstract class for vector database. A vector database is responsible for storing and retrieving documents.

    Attributes:
        active_collection: Any | The active collection in the vector database. Make get_collection faster. Default is None.
        type: str | The current vector database type. Default is "".
        embed_model: Any | The embedding model used for the vector database.
    """

    def create_collection(
        self, collection_name: str, overwrite: bool = False, _get_or_create: bool = True
    ) -> Any:
        """Create a collection in the vector database.
        Case 1. if the collection does not exist, create the collection.
        Case 2. the collection exists, if overwrite is True, it will overwrite the collection.
        Case 3. the collection exists and overwrite is False, if get_or_create is True, it will get the collection,
            otherwise it raise a ValueError.

        Args:
            collection_name: str | The name of the collection.
            overwrite: bool | Whether to overwrite the collection if it exists. Default is False.
            _get_or_create: bool | Whether to get the collection if it exists. Default is True.

        Returns:
            Any | The collection object.
        """
        ...

    def get_collection(self, collection_name: str | None = None) -> Any:
        """Get the collection from the vector database.

        Args:
            collection_name: str | The name of the collection. Default is None. If None, return the
                current active collection.

        Returns:
            Any | The collection object.
        """
        ...

    def get_collections(self) -> Any:
        """Get all the collections from the vector database.


        Returns:
            List[Any] | List of collection objects.
        """
        ...

    def delete_collection(self, collection_name: str) -> Any:
        """Delete the collection from the vector database.

        Args:
            collection_name: str | The name of the collection.

        Returns:
            Any
        """
        ...

    def insert_documents(
        self,
        docs: list[Document],
        collection_name: str | None = None,
        _upsert: bool = False,
        **kwargs,
    ) -> None:
        """Insert documents into the collection of the vector database.

        Args:
            docs: List[Document] | A list of documents. Each document is a TypedDict `Document`.
            collection_name: str | The name of the collection. Default is None.
            _upsert: bool | Whether to update the document if it exists. Default is False.
            kwargs: Dict | Additional keyword arguments.

        Returns:
            None
        """
        ...

    def update_documents(
        self, docs: list[Document], collection_name: str | None = None, **kwargs
    ) -> None:
        """Update documents in the collection of the vector database.

        Args:
            docs: List[Document] | A list of documents.
            collection_name: str | The name of the collection. Default is None.
            kwargs: Dict | Additional keyword arguments.

        Returns:
            None
        """
        ...

    def delete_documents(
        self, ids: list[ItemID], collection_name: str | None = None, **kwargs
    ) -> None:
        """Delete documents from the collection of the vector database.

        Args:
            ids: List[ItemID] | A list of document ids. Each id is a typed `ItemID`.
            collection_name: str | The name of the collection. Default is None.
            kwargs: Dict | Additional keyword arguments.

        Returns:
            None
        """
        ...

    def semantic_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        distance_threshold: float = -1,
        **kwargs: Any,
    ) -> QueryResults:
        """Retrieve documents from the collection of the vector database based on the queries.

        Args:
            queries: List[str] | A list of queries. Each query is a string.
            collection_name: str | The name of the collection. Default is None.
            n_results: int | The number of relevant documents to return. Default is 10.
            distance_threshold: float | The threshold for the distance score, only distance smaller than it will be
                returned. Don't filter with it if < 0. Default is -1.
            kwargs: Dict | Additional keyword arguments.

        Returns:
            QueryResults | The query results. Each query result is a list of list of tuples containing the document and
                the distance.
        """
        ...

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> QueryResults:
        """Search for documents in the collection of the vector database using lexical search.

        Args:
            queries: List[str] | A list of queries.
            collection_name: str | The name of the collection. Default is None.
            n_results: int | The number of results to return. Default is 10.
            kwargs: Dict | Additional keyword arguments.

        Returns:
            QueryResults | A list of query results.
        """
        ...

    def get_documents_by_ids(
        self,
        ids: list[ItemID] | None = None,
        collection_name: str | None = None,
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from the collection of the vector database based on the ids.

        Args:
            ids: List[ItemID] | A list of document ids. If None, will return all the documents. Default is None.
            collection_name: str | The name of the collection. Default is None.
            include: List[str] | The fields to include. Default is None.
                If None, will include ["metadatas", "documents"], ids will always be included. This may differ
                depending on the implementation.
            kwargs: dict | Additional keyword arguments.

        Returns:
            List[Document] | The results.
        """
        ...


class VectorDBFactory:
    """Factory class for creating vector databases."""

    DEFAULT_VECTOR_DB = "epistemic_graph"
    PREDEFINED_VECTOR_DB = [
        "epistemic_graph",
        "postgres",
        "mongodb",
        "qdrant",
    ]

    @staticmethod
    def create_vector_database(db_type: str | None = None, **kwargs) -> VectorDB:
        """Create a vector database.

        Args:
            db_type: str | The type of the vector database. When ``None``/empty it
                resolves to ``DEFAULT_VECTOR_DB`` (``epistemic_graph``).
            kwargs: Dict | The keyword arguments for initializing the vector database.

        Returns:
            VectorDB | The vector database.
        """
        db_type = str(db_type or VectorDBFactory.DEFAULT_VECTOR_DB).strip().casefold()
        if db_type == "epistemic_graph":
            from .epistemic_graph import EpistemicGraphVectorDB

            return EpistemicGraphVectorDB(**kwargs)
        if db_type == "postgres":
            from .postgres import PostgreSQL

            return PostgreSQL(**kwargs)
        if db_type == "mongodb":
            from .mongodb import MongoDBAtlasVectorDB

            return MongoDBAtlasVectorDB(**kwargs)
        if db_type == "qdrant":
            from .qdrant import QdrantVectorDB

            return QdrantVectorDB(**kwargs)
        raise ValueError("vector_database_type_unsupported")
