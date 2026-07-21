"""Local, policy-bound orchestration API for vector-mcp.

The MCP server is self-contained: it delegates storage to the selected vector
backend and obtains model, secret, and trust configuration exclusively through
``AgentConfig``.  No second HTTP API, model credential, or transport profile is
introduced here.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
import uuid
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from agent_utilities import create_embedding_model
from agent_utilities.core.config import setting
from agent_utilities.core.transport_security import (
    ResolvedTLSProfile,
    resolve_configured_tls_profile,
)
from agent_utilities.knowledge_graph.core.session import resolve_session
from agent_utilities.security.cli_secrets import (
    RuntimeSecretReferenceError,
    resolve_runtime_secret_reference,
)
from agent_utilities.security.guardrails import PiiSanitizer
from llama_index.core import SimpleDirectoryReader

from vector_mcp.backend_policy import ensure_backend_available
from vector_mcp.vectordb.base import Document, QueryResults, VectorDB, VectorDBFactory

_COLLECTION = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
_PRIVATE_METADATA = re.compile(
    r"(?:^|_)(?:file|filename|filepath|path|directory|dirname|uri|url|source)(?:$|_)",
    re.IGNORECASE,
)
_LOCAL_PATH = re.compile(
    r"(?<![\w:])(?:[A-Za-z]:[\\/][^\s<>\"']+|/(?:[^/\s]+/)+[^\s<>\"']*|\\\\[^\s\\]+\\[^\s]+)"
)
_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
_MAX_DOCUMENTS = 1_000
_MAX_DOCUMENT_TOTAL_BYTES = 512 * 1024 * 1024
_STABLE_ERROR = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class VectorConfigurationError(RuntimeError):
    """Stable configuration failure that does not disclose runtime values."""


def _configured(name: str, default: Any = None) -> Any:
    return setting(name, default)


def _required_text(name: str) -> str:
    value = str(_configured(name, "") or "").strip()
    if not value:
        raise VectorConfigurationError(f"{name.lower()}_required")
    return value


def _optional_secret(name: str) -> str | None:
    reference = str(_configured(name, "") or "").strip()
    if not reference:
        return None
    try:
        value = resolve_runtime_secret_reference(reference)
    except RuntimeSecretReferenceError:
        raise VectorConfigurationError(f"{name.lower()}_unavailable") from None
    if any(character in value for character in "\x00\r\n"):
        raise VectorConfigurationError(f"{name.lower()}_invalid")
    return value


def _required_secret(name: str) -> str:
    value = _optional_secret(name)
    if value is None:
        raise VectorConfigurationError(f"{name.lower()}_required")
    return value


def _tls(service: str) -> ResolvedTLSProfile:
    prefix = service.upper()
    try:
        profile = resolve_configured_tls_profile(
            service,
            profile_name=str(_configured(f"{prefix}_TLS_PROFILE", "") or "") or None,
            profile_ref=str(_configured(f"{prefix}_TLS_PROFILE_REF", "") or "") or None,
        )
    except Exception:
        raise VectorConfigurationError(f"{service}_tls_profile_invalid") from None
    if not profile.verify_enabled:
        raise VectorConfigurationError(f"{service}_tls_profile_invalid")
    return profile


def _safe_host(value: str) -> str:
    host = value.strip()
    if (
        not host
        or len(host) > 253
        or any(character.isspace() for character in host)
        or any(character in host for character in "/@?#\\")
    ):
        raise VectorConfigurationError("database_host_invalid")
    return host


def _port(default: int) -> int:
    try:
        value = int(_configured("DB_PORT", default))
    except (TypeError, ValueError):
        raise VectorConfigurationError("database_port_invalid") from None
    if not 1 <= value <= 65_535:
        raise VectorConfigurationError("database_port_invalid")
    return value


def _positive_int(name: str, default: int, maximum: int) -> int:
    try:
        value = int(_configured(name, default))
    except (TypeError, ValueError):
        raise VectorConfigurationError(f"{name.lower()}_invalid") from None
    if not 1 <= value <= maximum:
        raise VectorConfigurationError(f"{name.lower()}_invalid")
    return value


def _host_allowlist(name: str) -> list[str]:
    value = _configured(name, [])
    if not isinstance(value, (list, tuple)) or len(value) > 256:
        raise VectorConfigurationError(f"{name.lower()}_invalid")
    hosts: list[str] = []
    for item in value:
        try:
            hosts.append(_safe_host(str(item)))
        except VectorConfigurationError:
            raise VectorConfigurationError(f"{name.lower()}_invalid") from None
    return hosts


def _create_database(db_type: str, collection_name: str, embed_model: Any) -> VectorDB:
    """Build one backend from canonical AgentConfig fields only."""

    if db_type == "epistemic_graph":
        return VectorDBFactory.create_vector_database(
            db_type, embed_model=embed_model, collection_name=collection_name
        )
    if db_type == "postgres":
        return VectorDBFactory.create_vector_database(
            db_type,
            host=_safe_host(_required_text("DB_HOST")),
            port=_port(5432),
            dbname=_required_text("DBNAME"),
            username=_required_secret("DB_USERNAME_REF"),
            password=_required_secret("DB_PASSWORD_REF"),
            tls_profile=_tls("postgres"),
            timeout=_positive_int("POSTGRES_REQUEST_TIMEOUT", 30, 300),
            max_pool_size=_positive_int("POSTGRES_MAX_POOL_SIZE", 20, 100),
            embed_model=embed_model,
            collection_name=collection_name,
        )
    if db_type == "mongodb":
        return VectorDBFactory.create_vector_database(
            db_type,
            uri=_required_secret("MONGODB_URI_REF"),
            dbname=_required_text("DBNAME"),
            tls_profile=_tls("mongodb"),
            timeout_ms=_positive_int("MONGODB_REQUEST_TIMEOUT_MS", 30_000, 300_000),
            max_pool_size=_positive_int("MONGODB_MAX_POOL_SIZE", 20, 100),
            embed_model=embed_model,
            collection_name=collection_name,
        )
    if db_type == "qdrant":
        return VectorDBFactory.create_vector_database(
            db_type,
            host=_safe_host(_required_text("DB_HOST")),
            port=_port(6333),
            api_key=_required_secret("QDRANT_API_KEY_REF"),
            tls_profile=_tls("qdrant"),
            allowed_private_hosts=_host_allowlist("QDRANT_HTTP_ALLOWED_PRIVATE_HOSTS"),
            timeout=_positive_int("QDRANT_REQUEST_TIMEOUT", 30, 300),
            embed_model=embed_model,
            collection_name=collection_name,
        )
    raise VectorConfigurationError("vector_backend_unsupported")


class Api:
    """Functional vector API used directly by the MCP dependency boundary."""

    def __init__(self, *, embed_model: Any | None = None) -> None:
        self._embed_model = embed_model
        self._databases: dict[str, VectorDB] = {}
        self._sanitizer = PiiSanitizer()
        self._lock = RLock()

    @property
    def embed_model(self) -> Any:
        if self._embed_model is None:
            self._embed_model = create_embedding_model()
        return self._embed_model

    @staticmethod
    def _collection(value: str | None) -> str:
        if not value or _COLLECTION.fullmatch(value) is None:
            raise ValueError("collection_name_invalid")
        return value

    @staticmethod
    def _partition_prefix(required_scope: str) -> str:
        session = resolve_session(required_scope=required_scope)
        digest = hashlib.sha256(session.tenant.encode("utf-8")).hexdigest()[:16]
        return f"t_{digest}_"

    def _physical_collection(self, logical_name: str, required_scope: str) -> str:
        return f"{self._partition_prefix(required_scope)}{logical_name}"

    @staticmethod
    def _question(value: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or len(value.encode("utf-8")) > 1_048_576
        ):
            raise ValueError("search_question_invalid")
        return value

    @staticmethod
    def _limit(value: int) -> int:
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            raise ValueError("result_count_invalid") from None
        if isinstance(value, bool) or not 1 <= resolved <= 1_000:
            raise ValueError("result_count_invalid")
        return resolved

    def _database(self, db_type: str | None, collection_name: str) -> VectorDB:
        selected = ensure_backend_available(
            db_type
            or str(_configured("DATABASE_TYPE", "epistemic_graph") or "epistemic_graph")
        )
        with self._lock:
            database = self._databases.get(selected)
            if database is None:
                try:
                    database = _create_database(
                        selected, collection_name, self.embed_model
                    )
                except VectorConfigurationError:
                    raise
                except ImportError:
                    raise RuntimeError(
                        "vector_provider_dependency_unavailable"
                    ) from None
                except Exception:
                    raise RuntimeError("vector_backend_initialization_failed") from None
                self._databases[selected] = database
        return database

    @staticmethod
    def _invoke(operation: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except VectorConfigurationError:
            raise
        except ValueError as exc:
            code = str(exc)
            if _STABLE_ERROR.fullmatch(code):
                raise ValueError(code) from None
            raise RuntimeError("vector_backend_operation_failed") from None
        except ImportError:
            raise RuntimeError("vector_provider_dependency_unavailable") from None
        except Exception:
            raise RuntimeError("vector_backend_operation_failed") from None

    def _sanitize_text(self, value: str) -> str:
        return _LOCAL_PATH.sub("[REDACTED_PATH]", self._sanitizer.sanitize_text(value))

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                self._sanitize_text(str(key)): self._sanitize_value(item)
                for key, item in value.items()
                if _PRIVATE_METADATA.search(str(key)) is None
            }
        if isinstance(value, (list, tuple)):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, str):
            return self._sanitize_text(value)
        return value

    def _sanitize_metadata(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        return dict(self._sanitize_value(dict(metadata or {})))

    def _document(
        self, text: str, metadata: Mapping[str, Any] | None = None
    ) -> Document:
        if not isinstance(text, str):
            raise ValueError("document_content_invalid")
        encoded = text.encode("utf-8")
        if not encoded or len(encoded) > _MAX_DOCUMENT_BYTES:
            raise ValueError("document_content_invalid")
        content = self._sanitize_text(text)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return Document(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, digest)),
            content=content,
            metadata=self._sanitize_metadata(metadata),
        )

    def _load_documents(
        self,
        *,
        document_directory: Path | None = None,
        document_paths: list[Path] | None = None,
        document_contents: list[str] | None = None,
    ) -> list[Document]:
        if document_directory is not None and not isinstance(document_directory, Path):
            raise ValueError("resolved_document_input_required")
        if document_paths and any(
            not isinstance(value, Path) for value in document_paths
        ):
            raise ValueError("resolved_document_input_required")
        if len(document_contents or []) > _MAX_DOCUMENTS:
            raise ValueError("document_count_exceeded")
        total_bytes = sum(
            len(text.encode("utf-8"))
            for text in (document_contents or [])
            if isinstance(text, str)
        )
        if total_bytes > _MAX_DOCUMENT_TOTAL_BYTES:
            raise ValueError("document_total_size_exceeded")
        documents = [self._document(text) for text in (document_contents or [])]
        loaded: Iterable[Any] = ()
        try:
            if document_directory:
                loaded = SimpleDirectoryReader(input_dir=document_directory).load_data()
            elif document_paths:
                loaded = SimpleDirectoryReader(input_files=document_paths).load_data()
        except Exception:
            raise RuntimeError("document_loading_failed") from None
        for item in loaded:
            text = (
                item.get_content() if hasattr(item, "get_content") else str(item.text)
            )
            total_bytes += len(text.encode("utf-8"))
            if total_bytes > _MAX_DOCUMENT_TOTAL_BYTES:
                raise ValueError("document_total_size_exceeded")
            metadata = getattr(item, "metadata", None)
            documents.append(self._document(text, metadata))
            if len(documents) > _MAX_DOCUMENTS:
                raise ValueError("document_count_exceeded")
        if not documents:
            raise ValueError("document_input_required")
        unique: dict[str, Document] = {}
        for document in documents:
            unique[str(document["id"])] = document
        return list(unique.values())

    def _serialize(self, results: QueryResults) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for result_set in results:
            for document, score in result_set:
                if not math.isfinite(float(score)):
                    continue
                serialized.append(
                    {
                        "id": str(document.get("id", "")),
                        "content": self._sanitize_text(
                            str(document.get("content", ""))
                        ),
                        "metadata": self._sanitize_metadata(
                            document.get("metadata") or {}
                        ),
                        "score": float(score),
                    }
                )
        return serialized

    def create_collection(
        self,
        *,
        db_type: str | None = None,
        collection_name: str | None = None,
        overwrite: bool = False,
        document_directory: Path | None = None,
        document_paths: list[Path] | None = None,
        document_contents: list[str] | None = None,
    ) -> dict[str, Any]:
        name = self._collection(collection_name)
        physical = self._physical_collection(name, "kg:admin")
        database = self._database(db_type, physical)
        self._invoke(database.create_collection, physical, overwrite=bool(overwrite))
        count = 0
        if document_directory or document_paths or document_contents:
            documents = self._load_documents(
                document_directory=document_directory,
                document_paths=document_paths,
                document_contents=document_contents,
            )
            self._invoke(database.insert_documents, documents, physical, _upsert=True)
            count = len(documents)
        return {"status": "ready", "collection": name, "documents_added": count}

    def add_documents(
        self,
        *,
        db_type: str | None = None,
        collection_name: str | None = None,
        document_directory: Path | None = None,
        document_paths: list[Path] | None = None,
        document_contents: list[str] | None = None,
    ) -> dict[str, Any]:
        name = self._collection(collection_name)
        physical = self._physical_collection(name, "kg:write")
        documents = self._load_documents(
            document_directory=document_directory,
            document_paths=document_paths,
            document_contents=document_contents,
        )
        self._invoke(
            self._database(db_type, physical).insert_documents,
            documents,
            physical,
            _upsert=True,
        )
        return {
            "status": "updated",
            "collection": name,
            "documents_added": len(documents),
        }

    def delete_collection(
        self,
        *,
        db_type: str | None = None,
        collection_name: str | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if confirm is not True:
            raise ValueError("delete_confirmation_required")
        name = self._collection(collection_name)
        physical = self._physical_collection(name, "kg:admin")
        self._invoke(self._database(db_type, physical).delete_collection, physical)
        return {"status": "deleted", "collection": name}

    def list_collections(self, *, db_type: str | None = None) -> dict[str, Any]:
        prefix = self._partition_prefix("kg:admin")
        # A physical placeholder is used only to initialize a selected backend;
        # it is never created as part of listing.
        database = self._database(db_type, f"{prefix}memory")
        values = self._invoke(database.get_collections)
        names: list[str] = []
        for value in values:
            if isinstance(value, str):
                candidate = value
            elif isinstance(value, Mapping):
                candidate = value.get("name") or value.get("graph_name")
            elif getattr(value, "name", None):
                candidate = value.name
            else:
                candidate = None
            rendered = str(candidate or "")
            if rendered.startswith(prefix):
                names.append(rendered.removeprefix(prefix))
        return {
            "collections": [{"collection_name": name} for name in sorted(set(names))]
        }

    def semantic_search(
        self,
        *,
        db_type: str | None = None,
        collection_name: str | None = None,
        question: str,
        number_results: int = 10,
    ) -> dict[str, Any]:
        name = self._collection(collection_name)
        physical = self._physical_collection(name, "kg:read")
        question = self._question(question)
        number_results = self._limit(number_results)
        results = self._invoke(
            self._database(db_type, physical).semantic_search,
            [self._sanitize_text(question)],
            physical,
            int(number_results),
        )
        return {"results": self._sanitizer.sanitize(self._serialize(results))}

    def lexical_search(
        self,
        *,
        db_type: str | None = None,
        collection_name: str | None = None,
        question: str,
        number_results: int = 10,
    ) -> dict[str, Any]:
        name = self._collection(collection_name)
        physical = self._physical_collection(name, "kg:read")
        question = self._question(question)
        number_results = self._limit(number_results)
        results = self._invoke(
            self._database(db_type, physical).lexical_search,
            [self._sanitize_text(question)],
            physical,
            int(number_results),
        )
        return {"results": self._sanitizer.sanitize(self._serialize(results))}

    def search(
        self,
        *,
        db_type: str | None = None,
        collection_name: str | None = None,
        question: str,
        number_results: int = 10,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> dict[str, Any]:
        name = self._collection(collection_name)
        physical = self._physical_collection(name, "kg:read")
        question = self._question(question)
        limit = self._limit(number_results)
        rrf_is_bool = isinstance(rrf_k, bool)
        try:
            semantic_weight = float(semantic_weight)
            lexical_weight = float(lexical_weight)
            rrf_k = int(rrf_k)
        except (TypeError, ValueError):
            raise ValueError("search_parameters_invalid") from None
        if not math.isfinite(semantic_weight) or not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight_invalid")
        if not math.isfinite(lexical_weight) or not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight_invalid")
        if semantic_weight + lexical_weight <= 0:
            raise ValueError("search_weights_invalid")
        if rrf_is_bool or not 1 <= rrf_k <= 10_000:
            raise ValueError("rrf_k_invalid")
        safe_question = self._sanitize_text(question)
        database = self._database(db_type, physical)
        semantic = self._invoke(
            database.semantic_search, [safe_question], physical, limit
        )[0]
        lexical = self._invoke(
            database.lexical_search, [safe_question], physical, limit
        )[0]
        ranked: dict[str, tuple[Document, float]] = {}
        for weight, values in (
            (float(semantic_weight), semantic),
            (float(lexical_weight), lexical),
        ):
            for rank, (document, _score) in enumerate(values, start=1):
                identifier = str(document.get("id", ""))
                prior = ranked.get(identifier, (document, 0.0))[1]
                ranked[identifier] = (document, prior + weight / (int(rrf_k) + rank))
        fused = heapq.nlargest(limit, ranked.values(), key=lambda item: item[1])
        return {"results": self._sanitizer.sanitize(self._serialize([fused]))}


@lru_cache(maxsize=1)
def get_client() -> Api:
    """Return an embedded vector API; FastMCP owns its dependency lifetime."""

    return Api()
