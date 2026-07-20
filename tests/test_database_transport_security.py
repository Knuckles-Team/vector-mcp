from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vector_mcp.backend_policy import backend_status, ensure_backend_available
from vector_mcp.vector_api import VectorConfigurationError, _create_database


def _tls():
    return SimpleNamespace(
        verify_enabled=True,
        httpx_kwargs=lambda: {"verify": object(), "trust_env": False},
        pymongo_kwargs=lambda: {"tls": True},
        psycopg_kwargs=lambda: {"sslmode": "verify-full"},
    )


@pytest.mark.parametrize("name", ["epistemic_graph", "postgres", "mongodb", "qdrant"])
def test_only_current_backend_names_are_available(name: str) -> None:
    assert ensure_backend_available(name) == name
    assert backend_status(name)["available"] is True


@pytest.mark.parametrize(
    "removed", ["epistemic-graph", "eg", "chroma", "chromadb", "couchbase"]
)
def test_removed_or_legacy_backend_names_are_rejected(removed: str) -> None:
    with pytest.raises(ValueError, match="vector_backend_unsupported"):
        ensure_backend_available(removed)


def test_qdrant_builder_uses_secret_ref_and_shared_tls(monkeypatch) -> None:
    values = {
        "DB_HOST": "vector.example.invalid",
        "DB_PORT": 6333,
        "QDRANT_API_KEY_REF": "secret://runtime/qdrant",
        "QDRANT_REQUEST_TIMEOUT": 30,
    }
    factory = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        "vector_mcp.vector_api._configured",
        lambda name, default=None: values.get(name, default),
    )
    monkeypatch.setattr(
        "vector_mcp.vector_api.resolve_runtime_secret_reference",
        lambda reference: (
            "resolved" if reference == values["QDRANT_API_KEY_REF"] else None
        ),
    )
    monkeypatch.setattr("vector_mcp.vector_api._tls", lambda _service: _tls())
    monkeypatch.setattr(
        "vector_mcp.vector_api.VectorDBFactory.create_vector_database", factory
    )

    _create_database("qdrant", "docs", MagicMock())

    kwargs = factory.call_args.kwargs
    assert kwargs["host"] == "vector.example.invalid"
    assert kwargs["api_key"] == "resolved"
    assert kwargs["tls_profile"].verify_enabled is True


def test_postgres_requires_runtime_secret_references(monkeypatch) -> None:
    values = {
        "DB_HOST": "database.example.invalid",
        "DB_PORT": 5432,
        "DBNAME": "vectors",
        "DB_USERNAME_REF": "",
        "DB_PASSWORD_REF": "secret://runtime/password",
    }
    monkeypatch.setattr(
        "vector_mcp.vector_api._configured",
        lambda name, default=None: values.get(name, default),
    )
    with pytest.raises(VectorConfigurationError, match="db_username_ref_required"):
        _create_database("postgres", "docs", MagicMock())


def test_configuration_failures_do_not_echo_values(monkeypatch) -> None:
    secret_value = "do-not-disclose/path"
    monkeypatch.setattr(
        "vector_mcp.vector_api._configured",
        lambda name, default=None: secret_value if name == "DB_HOST" else default,
    )
    with pytest.raises(VectorConfigurationError) as exc_info:
        _create_database("qdrant", "docs", MagicMock())
    assert secret_value not in str(exc_info.value)
