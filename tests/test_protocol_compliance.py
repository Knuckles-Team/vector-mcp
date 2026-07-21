from __future__ import annotations

from vector_mcp.vectordb.base import VectorDBFactory


def test_factory_advertises_only_current_contract_backends() -> None:
    assert VectorDBFactory.PREDEFINED_VECTOR_DB == [
        "epistemic_graph",
        "postgres",
        "mongodb",
        "qdrant",
    ]
    assert VectorDBFactory.DEFAULT_VECTOR_DB == "epistemic_graph"


def test_factory_rejects_removed_backend() -> None:
    try:
        VectorDBFactory.create_vector_database("chroma")
    except ValueError as exc:
        assert str(exc) == "vector_database_type_unsupported"
    else:
        raise AssertionError("removed backend was accepted")
