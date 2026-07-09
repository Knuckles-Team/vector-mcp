#!/usr/bin/python
"""Epistemic-graph vector backend — the native, default RAG store.

Documents are stored as nodes in the local **epistemic-graph** engine (the
AI-native, redb-authoritative database): the chunk text + metadata live in node
properties and the embedding lives in the engine's native ANN index
(IVF-PQ / HNSW). ``semantic_search`` therefore runs as the engine's O(log N)
vector search over a durable, cross-process index — not an in-Python cosine scan.

A "collection" maps to an engine **graph** (``graph_name``); each collection is a
separately-connected :class:`SyncEpistemicGraphClient`. Embeddings are produced
client-side with the shared ``create_embedding_model`` (remote vLLM/OpenAI-style),
matching every other backend here.
"""

from typing import Any

from agent_utilities import create_embedding_model

from vector_mcp.vectordb.base import Document, ItemID, QueryResults, VectorDB
from vector_mcp.vectordb.db_utils import (
    get_logger,
    optional_import_block,
    require_optional_import,
)

with optional_import_block():
    from agent_utilities.core.config import setting
    from epistemic_graph.client import SyncEpistemicGraphClient

logger = get_logger(__name__)

# The chunk text is stored under the engine's standard ``description`` property: it is
# the field the engine keyword-indexes and that ``graph.discover`` hydrates, so lexical
# search runs engine-side (scalable) instead of an O(N) client scan. Everything not in
# the reserved set is treated as user metadata.
_TEXT_KEY = "description"
_RESERVED_KEYS = {_TEXT_KEY, "label", "id", "_similarity", "name", "type", "score"}


@require_optional_import(["epistemic_graph", "agent_utilities"], "epistemic-graph")
class EpistemicGraphVectorDB(VectorDB):
    """A vector database backed by the native epistemic-graph engine."""

    def __init__(
        self,
        *,
        embed_model: Any | None = None,
        collection_name: str = "memory",
        socket_path: str | None = None,
        auth_secret: str | None = None,
        tcp_addr: str | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize the vector database.

        Args:
            embed_model: Embedding model; defaults to ``create_embedding_model()``.
            collection_name: The engine graph a document set lives in.
            socket_path: UDS path to the engine (else ``GRAPH_SERVICE_SOCKET``).
            auth_secret: HMAC secret (else ``GRAPH_SERVICE_AUTH_SECRET``).
            tcp_addr: Optional ``host:port`` if the engine is reached over TCP.
        """
        self.embed_model = embed_model or create_embedding_model()
        self.collection_name = collection_name
        self.metadata = metadata or {}
        self.type = "epistemic-graph"
        self._socket_path = socket_path or setting("GRAPH_SERVICE_SOCKET", "") or None
        self._auth_secret = (
            auth_secret
            or setting("GRAPH_SERVICE_AUTH_SECRET", "")
            or setting("EPISTEMIC_GRAPH_SECRET", "")
            or None
        )
        self._tcp_addr = tcp_addr or setting("GRAPH_SERVICE_TCP_ADDR", "") or None
        self._clients: dict[str, Any] = {}
        self.active_collection = collection_name
        # Connect (and ensure) the default collection up front.
        self._client_for(collection_name)

    # ── connection ────────────────────────────────────────────────────────
    def _client_for(self, collection_name: str | None = None) -> Any:
        name = collection_name or self.collection_name
        client = self._clients.get(name)
        if client is None:
            connect_kwargs: dict[str, Any] = {"graph_name": name}
            if self._socket_path:
                connect_kwargs["socket_path"] = self._socket_path
            if self._tcp_addr:
                connect_kwargs["tcp_addr"] = self._tcp_addr
            if self._auth_secret:
                connect_kwargs["auth_secret"] = self._auth_secret
            client = SyncEpistemicGraphClient.connect(**connect_kwargs)
            try:
                client.tenants.create(name)
            except Exception as e:  # graph may already exist — non-fatal
                logger.debug(f"tenant create {name}: {e}")
            self._clients[name] = client
        return client

    def _embed(self, text: str, *, is_query: bool = False) -> list[float]:
        vec = (
            self.embed_model.get_query_embedding(text)
            if is_query
            else self.embed_model.get_text_embedding(text)
        )
        return [float(x) for x in vec]

    @staticmethod
    def _props_to_doc(node_id: str, props: dict | None) -> Document:
        props = dict(props or {})
        content = props.get(_TEXT_KEY, "") or ""
        meta = {k: v for k, v in props.items() if k not in _RESERVED_KEYS}
        return Document(id=node_id, content=content, metadata=meta, embedding=None)

    # ── collections ───────────────────────────────────────────────────────
    def create_collection(
        self, collection_name: str, overwrite: bool = False, _get_or_create: bool = True
    ) -> Any:
        self.collection_name = collection_name
        client = self._client_for(collection_name)
        if overwrite:
            try:
                client.graph.clear()
            except Exception as e:
                logger.info(f"clear collection {collection_name}: {e}")
        self.active_collection = collection_name
        return collection_name

    def get_collection(self, collection_name: str | None = None) -> Any:
        return collection_name or self.collection_name

    def get_collections(self) -> Any:
        client = self._client_for(self.collection_name)
        try:
            graphs = client.tenants.list() or []
        except Exception as e:
            logger.info(f"list collections: {e}")
            return list(self._clients)
        names = []
        for g in graphs:
            if isinstance(g, dict):
                names.append(g.get("graph_name") or g.get("name"))
            else:
                names.append(g)
        return [n for n in names if n]

    def delete_collection(self, collection_name: str) -> None:
        client = self._client_for(collection_name)
        try:
            client.tenants.delete(collection_name)
        except Exception as e:
            logger.info(f"delete collection {collection_name}: {e}")
        self._clients.pop(collection_name, None)
        if self.active_collection == collection_name:
            self.active_collection = ""

    # ── documents ─────────────────────────────────────────────────────────
    def insert_documents(
        self,
        docs: list[Document],
        collection_name: str | None = None,
        _upsert: bool = False,
        **kwargs,
    ) -> None:
        client = self._client_for(collection_name)
        for doc in docs:
            doc_id = str(doc["id"])
            text = doc.get("content") or ""
            meta = {
                k: v
                for k, v in (doc.get("metadata") or {}).items()
                if k not in _RESERVED_KEYS
            }
            embedding = doc.get("embedding")
            if embedding is None:
                embedding = self._embed(text)
            client.nodes.add(doc_id, {_TEXT_KEY: text, "label": "Document", **meta})
            client.graph.add_embedding(doc_id, [float(x) for x in embedding])

    def update_documents(
        self, docs: list[Document], collection_name: str | None = None, **kwargs
    ) -> None:
        # Node add is an upsert on the engine (same id overwrites properties).
        self.insert_documents(docs, collection_name, _upsert=True, **kwargs)

    def delete_documents(
        self, ids: list[ItemID], collection_name: str | None = None, **kwargs
    ) -> None:
        client = self._client_for(collection_name)
        for _id in ids:
            try:
                client.nodes.remove(str(_id))
            except Exception as e:
                logger.info(f"delete node {_id}: {e}")

    def get_documents_by_ids(
        self,
        ids: list[ItemID] | None = None,
        collection_name: str | None = None,
        include: list[str] | None = None,
        **kwargs,
    ) -> list[Document]:
        client = self._client_for(collection_name)
        docs: list[Document] = []
        if ids:
            for _id in ids:
                try:
                    props = client.nodes.properties(str(_id))
                except Exception:
                    props = None
                if props is not None:
                    docs.append(self._props_to_doc(str(_id), props))
        else:
            try:
                listing = client.nodes.list() or []
            except Exception as e:
                logger.info(f"list nodes: {e}")
                listing = []
            for entry in listing:
                node_id = str(entry[0]) if isinstance(entry, list | tuple) else str(entry)
                try:
                    props = client.nodes.properties(node_id)
                except Exception:
                    props = None
                docs.append(self._props_to_doc(node_id, props))
        return docs

    # ── search ────────────────────────────────────────────────────────────
    def semantic_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        distance_threshold: float = -1,
        **kwargs: Any,
    ) -> QueryResults:
        client = self._client_for(collection_name)
        results: QueryResults = []
        for query in queries:
            q_emb = self._embed(query, is_query=True)
            try:
                hits = client.graph.semantic_search(q_emb, n_results) or []
            except Exception as e:
                logger.error(f"epistemic-graph semantic_search failed: {e}")
                hits = []
            query_result: list[tuple[Document, float]] = []
            for item in hits:
                if isinstance(item, list | tuple) and len(item) >= 2:
                    node_id, score = str(item[0]), float(item[1])
                elif isinstance(item, dict):
                    node_id = str(item.get("id", ""))
                    score = float(item.get("_similarity", item.get("score", 0.0)) or 0.0)
                else:
                    continue
                if not node_id:
                    continue
                distance = 1.0 - score
                if distance_threshold >= 0 and distance > distance_threshold:
                    continue
                try:
                    props = client.nodes.properties(node_id)
                except Exception:
                    props = None
                query_result.append((self._props_to_doc(node_id, props), distance))
            results.append(query_result)
        return results

    def lexical_search(
        self,
        queries: list[str],
        collection_name: str | None = None,
        n_results: int = 10,
        **kwargs: Any,
    ) -> QueryResults:
        """Keyword search over the engine's scalable index.

        Routes through the engine's one-round-trip ``discover`` op — it ranks nodes by
        keyword overlap over the indexed text (the chunk text lives in ``description``)
        AND semantic similarity, returning a hydrated top-k. Falls back to a client-side
        term scan only if the engine is too old to support ``discover``.
        """
        client = self._client_for(collection_name)
        results: QueryResults = []
        for query in queries:
            terms = list(dict.fromkeys(t for t in query.lower().split() if t))
            try:
                q_emb = self._embed(query, is_query=True)
            except Exception:
                q_emb = []  # discover degrades to a bounded keyword-only scan
            try:
                hits = client.graph.discover(terms, q_emb, n_results) or []
            except Exception as e:
                logger.info(f"discover unavailable, using client scan: {e}")
                results.append(self._lexical_scan(client, terms, n_results))
                continue
            query_result: list[tuple[Document, float]] = []
            for item in hits[:n_results]:
                if isinstance(item, dict):
                    node_id = str(item.get("id", ""))
                    score = float(item.get("score", 0.0) or 0.0)
                    # discover hydrates text in-band; fetch full props for metadata.
                    try:
                        props = client.nodes.properties(node_id) or item
                    except Exception:
                        props = item
                else:
                    node_id, score, props = str(item), 0.0, None
                if not node_id:
                    continue
                query_result.append((self._props_to_doc(node_id, props), score))
            results.append(query_result)
        return results

    def _lexical_scan(
        self, client: Any, terms: list[str], n_results: int
    ) -> list[tuple[Document, float]]:
        """Fallback term-frequency scan (only when the engine lacks ``discover``)."""
        try:
            listing = client.nodes.list() or []
        except Exception as e:
            logger.info(f"lexical scan (list nodes): {e}")
            return []
        scored: list[tuple[Document, float]] = []
        for entry in listing:
            node_id = str(entry[0]) if isinstance(entry, list | tuple) else str(entry)
            try:
                props = client.nodes.properties(node_id) or {}
            except Exception:
                continue
            content = str(props.get(_TEXT_KEY, "") or "").lower()
            if not content:
                continue
            freq = sum(content.count(t) for t in terms)
            if freq > 0:
                scored.append((self._props_to_doc(node_id, props), float(freq)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n_results]
