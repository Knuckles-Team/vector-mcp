from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vector_mcp.vector_api import Api, get_client


def _api_with_database(monkeypatch):
    database = MagicMock()
    monkeypatch.setattr(
        Api, "_partition_prefix", staticmethod(lambda _scope: "t_partition_")
    )
    monkeypatch.setattr(
        "vector_mcp.vector_api._create_database", lambda *_args, **_kwargs: database
    )
    return Api(embed_model=MagicMock()), database


def test_collection_partition_is_hashed_from_verified_tenant(monkeypatch) -> None:
    session = SimpleNamespace(tenant="tenant-authority")
    resolver = MagicMock(return_value=session)
    monkeypatch.setattr("vector_mcp.vector_api.resolve_session", resolver)
    api = Api(embed_model=MagicMock())

    physical = api._physical_collection("knowledge", "kg:write")

    resolver.assert_called_once_with(required_scope="kg:write")
    assert physical.endswith("_knowledge")
    assert "tenant-authority" not in physical


def test_get_client_is_embedded_and_needs_no_duplicate_model_endpoint() -> None:
    assert isinstance(get_client(), Api)


def test_create_sanitizes_pii_and_does_not_store_origin_metadata(monkeypatch) -> None:
    api, database = _api_with_database(monkeypatch)

    response = api.create_collection(
        db_type="epistemic_graph",
        collection_name="knowledge",
        document_contents=["Contact analyst@example.invalid for access."],
    )

    assert response == {
        "status": "ready",
        "collection": "knowledge",
        "documents_added": 1,
    }
    stored = database.insert_documents.call_args.args[0][0]
    assert "analyst@example.invalid" not in stored["content"]
    assert stored["metadata"] == {}
    assert "path" not in str(stored).casefold()


def test_loaded_document_origin_fields_are_removed(monkeypatch) -> None:
    api, _database = _api_with_database(monkeypatch)
    item = SimpleNamespace(
        metadata={"file_path": "/private/location", "topic": "safe"},
        get_content=lambda: "public body",
    )
    reader = MagicMock()
    reader.load_data.return_value = [item]
    monkeypatch.setattr(
        "vector_mcp.vector_api.SimpleDirectoryReader", MagicMock(return_value=reader)
    )

    document = api._load_documents(document_paths=[Path("approved-input")])[0]

    assert document["metadata"] == {"topic": "safe"}


def test_content_and_nested_metadata_do_not_retain_host_paths() -> None:
    api = Api(embed_model=MagicMock())

    document = api._document(
        "Read /private/workspace/report.md",
        {"details": {"location": "/private/workspace/input.txt"}},
    )

    assert "/private/" not in document["content"]
    assert "/private/" not in str(document["metadata"])


def test_api_rejects_unresolved_filesystem_strings() -> None:
    api = Api(embed_model=MagicMock())
    with pytest.raises(ValueError, match="resolved_document_input_required"):
        api._load_documents(document_paths=["/untrusted/location"])  # type: ignore[list-item]


def test_inline_ingestion_is_total_bounded_and_content_deduplicated(
    monkeypatch,
) -> None:
    api = Api(embed_model=MagicMock())
    assert len(api._load_documents(document_contents=["same", "same"])) == 1

    monkeypatch.setattr("vector_mcp.vector_api._MAX_DOCUMENT_TOTAL_BYTES", 3)
    with pytest.raises(ValueError, match="document_total_size_exceeded"):
        api._load_documents(document_contents=["bounded"])


def test_delete_requires_explicit_confirmation(monkeypatch) -> None:
    api, database = _api_with_database(monkeypatch)
    with pytest.raises(ValueError, match="delete_confirmation_required"):
        api.delete_collection(
            db_type="epistemic_graph", collection_name="knowledge", confirm=False
        )
    database.delete_collection.assert_not_called()


def test_provider_errors_are_replaced_with_path_and_secret_safe_codes(
    monkeypatch,
) -> None:
    api, database = _api_with_database(monkeypatch)
    database.get_collections.side_effect = RuntimeError(
        "connection failed for credential@host/private/location"
    )

    with pytest.raises(RuntimeError, match="^vector_backend_operation_failed$") as exc:
        api.list_collections(db_type="epistemic_graph")

    assert "credential" not in str(exc.value)


def test_collection_inventory_returns_only_current_tenant_logical_names(
    monkeypatch,
) -> None:
    api, database = _api_with_database(monkeypatch)
    database.get_collections.return_value = [
        "t_partition_alpha",
        "t_other_hidden",
        "t_partition_beta",
    ]

    assert api.list_collections(db_type="epistemic_graph") == {
        "collections": [
            {"collection_name": "alpha"},
            {"collection_name": "beta"},
        ]
    }


def test_hybrid_search_uses_bounded_rrf_without_duplicate_documents(
    monkeypatch,
) -> None:
    api, database = _api_with_database(monkeypatch)
    first = {"id": "a", "content": "first", "metadata": {}}
    second = {"id": "b", "content": "second", "metadata": {}}
    database.semantic_search.return_value = [[(first, 0.9), (second, 0.8)]]
    database.lexical_search.return_value = [[(second, 2.0), (first, 1.0)]]

    response = api.search(
        db_type="epistemic_graph",
        collection_name="knowledge",
        question="topic",
        number_results=2,
        semantic_weight=0.5,
        lexical_weight=0.5,
        rrf_k=60,
    )

    assert {item["id"] for item in response["results"]} == {"a", "b"}
    assert len(response["results"]) == 2
