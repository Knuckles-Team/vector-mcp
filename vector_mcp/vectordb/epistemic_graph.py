#!/usr/bin/python
"""Epistemic-graph vector backend — the native, default RAG store.

Documents are stored as nodes in the local **epistemic-graph** engine (the
AI-native, redb-authoritative database): the chunk text + metadata live in node
properties and the embedding lives in the engine's native ANN index
(IVF-PQ / HNSW). ``semantic_search`` therefore runs as the engine's O(log N)
vector search over a durable, cross-process index — not an in-Python cosine scan.

A "collection" maps to an engine **graph** (``graph_name``); each collection is a
separately-connected :class:`SyncEpistemicGraphClient`, cached per verified
authority so a session change never reuses another caller's connection. Engine
location, authentication, autostart, and deployment mode are owned entirely by
AgentConfig/the current engine client — this module never accepts or persists
transport or credential material. Embeddings are produced client-side with the
shared ``create_embedding_model`` (remote vLLM/OpenAI-style), matching every
other backend here.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from agent_utilities import create_embedding_model
from agent_utilities.knowledge_graph.core.session import resolve_session

from vector_mcp.vectordb.base import (
    Document,
    ItemID,
    QueryResults,
    VectorDB,
    document_embeddings,
)
from vector_mcp.vectordb.db_utils import (
    get_logger,
    optional_import_block,
    require_optional_import,
)

with optional_import_block():
    from epistemic_graph.client import SyncEpistemicGraphClient

logger = get_logger(__name__)

# The chunk text is stored under the engine's standard ``description`` property: it is
# the field the engine keyword-indexes and that ``graph.discover`` hydrates, so lexical
# search runs engine-side (scalable) instead of an O(N) client scan. Everything not in
# the reserved set is treated as user metadata.
_LABEL = "VectorDocument"
_TEXT_KEY = "description"
_RESERVED = {_TEXT_KEY, "name", "type", "label", "id", "score", "_similarity"}
_WORD = re.compile(r"[A-Za-z0-9_]{2,}")


@require_optional_import(["epistemic_graph", "agent_utilities"], "epistemic-graph")
class EpistemicGraphVectorDB(VectorDB):
    """A vector database backed by the native epistemic-graph engine.

    Client connections are O(1)-cached per ``(collection, authority)`` and every
    graph/node/txn operation rides the caller's verified :class:`GraphSession` —
    there is no constructor path to target an arbitrary socket/secret.
    """

    def __init__(
        self,
        *,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.embed_model = embed_model or create_embedding_model()
        self.collection_name = collection_name
        self.metadata = metadata or {}
        self.type = "epistemic_graph"
        self.active_collection = collection_name
        self._clients: dict[tuple[str, str], Any] = {}

    # ── connection ────────────────────────────────────────────────────────
    def _client_for(
        self, collection_name: str | None = None, *, session: Any | None = None
    ) -> Any:
        name = collection_name or self.collection_name
        session = session or resolve_session()
        verified_context = session.engine_verified_context()
        authority_digest = hashlib.sha256(
            json.dumps(verified_context, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        cache_key = (name, authority_digest)
        client = self._clients.get(cache_key)
        if client is None:
            # Engine location, authentication, autostart, and deployment mode are
            # owned entirely by AgentConfig/the current engine client.
            client = SyncEpistemicGraphClient.connect(
                graph_name=name, verified_context=verified_context
            )
            self._clients[cache_key] = client
        return client

    def _embedding(self, text: str, *, query: bool = False) -> list[float]:
        vector = (
            self.embed_model.get_query_embedding(text)
            if query
            else self.embed_model.get_text_embedding(text)
        )
        return [float(value) for value in vector]

    @staticmethod
    def _document(node_id: str, properties: dict[str, Any] | None) -> Document:
        values = dict(properties or {})
        return Document(
            id=node_id,
            content=str(values.get(_TEXT_KEY, "") or ""),
            metadata={
                key: value for key, value in values.items() if key not in _RESERVED
            },
        )

    @staticmethod
    def _graph_names(values: list[Any]) -> set[str]:
        names: set[str] = set()
        for value in values:
            if isinstance(value, str):
                names.add(value)
            elif isinstance(value, dict):
                candidate = value.get("name") or value.get("graph_name")
                if candidate:
                    names.add(str(candidate))
            elif isinstance(value, (tuple, list)) and value:
                names.add(str(value[0]))
        return names

    def _control_client(self) -> Any:
        session = resolve_session()
        return self._client_for(
            str(getattr(session, "graph", "") or "__commons__"), session=session
        )

    # ── collections ───────────────────────────────────────────────────────
    def create_collection(
        self, collection_name: str, overwrite: bool = False, _get_or_create: bool = True
    ) -> Any:
        control = self._control_client()
        exists = collection_name in self._graph_names(control.tenants.list())
        cached_clients: list[Any] = []
        if exists and overwrite:
            cached_clients = [
                self._clients.pop(key)
                for key in tuple(self._clients)
                if key[0] == collection_name
            ]
            control.tenants.delete(collection_name)
            exists = False
        if exists and not _get_or_create:
            raise ValueError("collection_exists")
        if not exists:
            control.tenants.create(collection_name, "Agent")
        if overwrite:
            for cached in cached_clients:
                cached.close()
        self.collection_name = collection_name
        self.active_collection = collection_name
        return self._client_for(collection_name)

    def get_collection(self, collection_name: str | None = None) -> Any:
        return self._client_for(collection_name)

    def get_collections(self) -> list[str]:
        return sorted(self._graph_names(self._control_client().tenants.list()))

    def delete_collection(self, collection_name: str) -> None:
        control = self._control_client()
        cached_clients = [
            self._clients.pop(key)
            for key in tuple(self._clients)
            if key[0] == collection_name
        ]
        control.tenants.delete(collection_name)
        for cached in cached_clients:
            cached.close()
        if self.active_collection == collection_name:
            self.active_collection = ""

    # ── documents ─────────────────────────────────────────────────────────
    def insert_documents(
        self,
        docs: list[Document],
        collection_name: str | None = None,
        _upsert: bool = False,
        **_kwargs: Any,
    ) -> None:
        client = self._client_for(collection_name)
        identifiers = [str(document["id"]) for document in docs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("document_ids_duplicate")
        if not _upsert:
            existing = client.nodes.has_batch(identifiers)
            if any(existing.get(identifier, False) for identifier in identifiers):
                raise ValueError("document_exists")
        vectors = document_embeddings(docs, self.embed_model)
        transaction = client.txn.begin()
        try:
            for document, identifier, vector in zip(
                docs, identifiers, vectors, strict=True
            ):
                content = str(document["content"])
                properties = {
                    **dict(document.get("metadata") or {}),
                    _TEXT_KEY: content,
                    "name": identifier,
                    "type": _LABEL,
                    "label": _LABEL,
                }
                client.txn.add_node(transaction, identifier, properties)
                client.txn.add_embedding(transaction, identifier, vector)
            if not client.txn.commit(transaction):
                raise ValueError("transaction_conflict")
        except Exception:
            try:
                client.txn.rollback(transaction)
            except Exception:
                pass
            raise

    def update_documents(
        self, docs: list[Document], collection_name: str | None = None, **kwargs: Any
    ) -> None:
        self.insert_documents(docs, collection_name, _upsert=True, **kwargs)

    def delete_documents(
        self, ids: list[ItemID], collection_name: str | None = None, **_kwargs: Any
    ) -> None:
        client = self._client_for(collection_name)
        transaction = client.txn.begin()
        try:
            for identifier in ids:
                client.txn.remove_node(transaction, str(identifier))
            if not client.txn.commit(transaction):
                raise ValueError("transaction_conflict")
        except Exception:
            try:
                client.txn.rollback(transaction)
            except Exception:
                pass
            raise

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
        client = self._client_for(collection_name)
        identifiers = [str(value) for value in ids]
        properties = client.nodes.properties_batch(identifiers)
        return [
            self._document(identifier, properties.get(identifier))
            for identifier in identifiers
            if properties.get(identifier) is not None
        ]

    # ── search ────────────────────────────────────────────────────────────
    def semantic_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        distance_threshold: float = -1,
        **_kwargs: Any,
    ) -> QueryResults:
        client = self._client_for(collection_name)
        output: QueryResults = []
        for query in queries:
            hits = client.graph.semantic_search(
                self._embedding(query, query=True), n_results
            )
            identifiers = [str(identifier) for identifier, _score in hits]
            properties = (
                client.nodes.properties_batch(identifiers) if identifiers else {}
            )
            values: list[tuple[Document, float]] = []
            for identifier, score in hits:
                similarity = float(score)
                if distance_threshold >= 0 and 1.0 - similarity > distance_threshold:
                    continue
                values.append(
                    (
                        self._document(
                            str(identifier), properties.get(str(identifier))
                        ),
                        similarity,
                    )
                )
            output.append(values)
        return output

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        **_kwargs: Any,
    ) -> QueryResults:
        """Keyword search over the engine's scalable index.

        Routes through the engine's one-round-trip ``discover`` op — it ranks nodes by
        keyword overlap over the indexed text (the chunk text lives in ``description``)
        AND semantic similarity, returning a hydrated top-k with no N+1 metadata fetch.
        Falls back to a bounded client-side term scan only if the engine is too old to
        support ``discover``.
        """
        client = self._client_for(collection_name)
        output: QueryResults = []
        for query in queries:
            keywords = list(
                dict.fromkeys(word.casefold() for word in _WORD.findall(query))
            )
            if not keywords:
                output.append([])
                continue
            try:
                embedding = self._embedding(query, query=True)
            except Exception:
                embedding = []  # discover degrades to a bounded keyword-only scan
            try:
                hits = client.graph.discover(keywords, embedding, n_results) or []
            except Exception as exc:
                logger.info(f"discover unavailable, using client scan: {exc}")
                output.append(self._lexical_scan(client, keywords, n_results))
                continue
            values: list[tuple[Document, float]] = []
            for hit in hits[:n_results]:
                properties = dict(hit)
                identifier = str(properties.pop("id", "") or "")
                if not identifier:
                    continue
                values.append(
                    (self._document(identifier, properties), float(hit.get("score", 0.0) or 0.0))
                )
            output.append(values)
        return output

    def _lexical_scan(
        self, client: Any, keywords: list[str], n_results: int
    ) -> list[tuple[Document, float]]:
        """Fallback term-frequency scan (only when the engine lacks ``discover``)."""
        try:
            listing = client.nodes.list() or []
        except Exception as exc:
            logger.info(f"lexical scan (list nodes): {exc}")
            return []
        scored: list[tuple[Document, float]] = []
        for entry in listing:
            node_id = str(entry[0]) if isinstance(entry, (list, tuple)) else str(entry)
            try:
                properties = client.nodes.properties(node_id) or {}
            except Exception:
                continue
            content = str(properties.get(_TEXT_KEY, "") or "").casefold()
            if not content:
                continue
            freq = sum(content.count(term) for term in keywords)
            if freq > 0:
                scored.append((self._document(node_id, properties), float(freq)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:n_results]

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
