"""Secure Qdrant backend using indexed vector and full-text operations."""

from __future__ import annotations

import ipaddress
import uuid
from functools import cached_property
from typing import Any

from agent_utilities import create_embedding_model
from agent_utilities.core.http_client import pinned_egress_transport
from agent_utilities.core.transport_security import ResolvedTLSProfile

from vector_mcp.vectordb.base import (
    Document,
    ItemID,
    QueryResults,
    VectorDB,
    document_embeddings,
)
from vector_mcp.vectordb.db_utils import optional_import_block, require_optional_import

with optional_import_block():
    from qdrant_client import QdrantClient, models

_POINT_NAMESPACE = uuid.UUID("9c4de92d-ec1b-4c67-8a45-bc359b0f4cea")


def _url(host: str, port: int) -> str:
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return f"https://{host}:{port}"
    rendered = f"[{parsed}]" if parsed.version == 6 else str(parsed)
    return f"https://{rendered}:{port}"


def _point_id(identifier: ItemID) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, str(identifier)))


@require_optional_import(["qdrant_client"], "qdrant")
class QdrantVectorDB(VectorDB):
    """Qdrant provider with shared TLS and no local/plaintext fallback."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        tls_profile: ResolvedTLSProfile,
        api_key: str,
        allowed_private_hosts: list[str] | tuple[str, ...] = (),
        timeout: int = 30,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        if not tls_profile.verify_enabled:
            raise ValueError("qdrant_tls_profile_invalid")
        if not api_key:
            raise ValueError("qdrant_api_key_required")
        self.collection_name = collection_name
        self.embed_model = embed_model or create_embedding_model()
        self.metadata = metadata or {}
        self.type = "qdrant"
        self.active_collection = collection_name
        httpx_options = dict(tls_profile.httpx_kwargs())
        verify = httpx_options.pop("verify", True)
        httpx_options.pop("trust_env", None)
        if httpx_options.pop("proxy", None) is not None:
            raise ValueError("qdrant_proxy_unsupported")
        transport = pinned_egress_transport(
            verify=verify,
            allowed_private_hosts=allowed_private_hosts,
            allow_loopback=False,
        )
        self.client = QdrantClient(
            url=_url(host, int(port)),
            api_key=api_key,
            timeout=int(timeout),
            prefer_grpc=False,
            # The SDK's compatibility probe creates an independent HTTP client
            # outside the injected TLS/DNS-pinned transport. Normal API calls
            # surface protocol incompatibility through the governed client.
            check_compatibility=False,
            verify=verify,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
            **httpx_options,
        )

    @cached_property
    def _embedding_dimension(self) -> int:
        return len(self.embed_model.get_query_embedding("vector dimension probe"))

    def _dimension(self) -> int:
        return self._embedding_dimension

    @staticmethod
    def _document(point: Any) -> Document:
        payload = dict(point.payload or {})
        return Document(
            id=str(payload.get("document_id", "")),
            content=str(payload.get("content", "")),
            metadata=dict(payload.get("metadata") or {}),
        )

    def create_collection(
        self, collection_name: str, overwrite: bool = False, _get_or_create: bool = True
    ) -> Any:
        exists = self.client.collection_exists(collection_name)
        if exists and overwrite:
            self.client.delete_collection(collection_name)
            exists = False
        if exists and not _get_or_create:
            raise ValueError("collection_exists")
        dimension = self._dimension()
        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )
        else:
            info = self.client.get_collection(collection_name)
            vectors = info.config.params.vectors
            if (
                getattr(vectors, "size", None) != dimension
                or getattr(vectors, "distance", None) != models.Distance.COSINE
            ):
                raise ValueError("collection_vector_schema_mismatch")
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="content",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
            ),
            wait=True,
        )
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        self.collection_name = collection_name
        self.active_collection = collection_name
        return self.client.get_collection(collection_name)

    def get_collection(self, collection_name: str | None = None) -> Any:
        return self.client.get_collection(collection_name or self.collection_name)

    def get_collections(self) -> list[str]:
        response = self.client.get_collections()
        return [str(item.name) for item in response.collections]

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name)
        if self.active_collection == collection_name:
            self.active_collection = ""

    def insert_documents(
        self,
        docs: list[Document],
        collection_name: str | None = None,
        _upsert: bool = False,
        **_kwargs: Any,
    ) -> None:
        name = collection_name or self.collection_name
        points = []
        identifiers = [str(document["id"]) for document in docs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("document_ids_duplicate")
        vectors = document_embeddings(docs, self.embed_model)
        if not _upsert and identifiers:
            existing = self.client.retrieve(
                name,
                [_point_id(identifier) for identifier in identifiers],
                with_payload=False,
                with_vectors=False,
            )
            if existing:
                raise ValueError("document_exists")
        for document, identifier, vector in zip(
            docs, identifiers, vectors, strict=True
        ):
            content = str(document["content"])
            points.append(
                models.PointStruct(
                    id=_point_id(identifier),
                    vector=vector,
                    payload={
                        "document_id": identifier,
                        "content": content,
                        "metadata": dict(document.get("metadata") or {}),
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=name, points=points, wait=True)

    def update_documents(
        self, docs: list[Document], collection_name: str | None = None, **kwargs: Any
    ) -> None:
        self.insert_documents(docs, collection_name, _upsert=True, **kwargs)

    def delete_documents(
        self, ids: list[ItemID], collection_name: str | None = None, **_kwargs: Any
    ) -> None:
        self.client.delete(
            collection_name=collection_name or self.collection_name,
            points_selector=models.PointIdsList(
                points=[_point_id(identifier) for identifier in ids]
            ),
            wait=True,
        )

    def get_documents_by_ids(
        self,
        ids: list[ItemID] | None = None,
        collection_name: str | None = None,
        include: list[str] | None = None,
        **_kwargs: Any,
    ) -> list[Document]:
        del include
        if not ids:
            raise ValueError("document_ids_required")
        points = self.client.retrieve(
            collection_name=collection_name or self.collection_name,
            ids=[_point_id(identifier) for identifier in ids],
            with_payload=True,
            with_vectors=False,
        )
        return [self._document(point) for point in points]

    def semantic_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        distance_threshold: float = -1,
        **_kwargs: Any,
    ) -> QueryResults:
        name = collection_name or self.collection_name
        results: QueryResults = []
        for query in queries:
            response = self.client.query_points(
                collection_name=name,
                query=[
                    float(value)
                    for value in self.embed_model.get_query_embedding(query)
                ],
                limit=int(n_results),
                with_payload=True,
                with_vectors=False,
            )
            values: list[tuple[Document, float]] = []
            for point in response.points:
                similarity = float(point.score)
                if distance_threshold >= 0 and 1.0 - similarity > distance_threshold:
                    continue
                values.append((self._document(point), similarity))
            results.append(values)
        return results

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        **_kwargs: Any,
    ) -> QueryResults:
        name = collection_name or self.collection_name
        results: QueryResults = []
        for query in queries:
            points, _offset = self.client.scroll(
                collection_name=name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content", match=models.MatchText(text=query)
                        )
                    ]
                ),
                limit=int(n_results),
                with_payload=True,
                with_vectors=False,
            )
            results.append(
                [
                    (self._document(point), 1.0 / rank)
                    for rank, point in enumerate(points, start=1)
                ]
            )
        return results

    def close(self) -> None:
        self.client.close()
