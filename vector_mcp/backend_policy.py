"""Fail-closed availability policy for vector backends."""

from __future__ import annotations

from typing import Final, TypedDict

AVAILABLE_BACKENDS: Final = frozenset(
    {"epistemic_graph", "mongodb", "postgres", "qdrant"}
)


class BackendStatus(TypedDict):
    backend: str
    available: bool
    reason: str
    advisory: str | None


def canonical_backend(value: str | None) -> str:
    return (value or "").strip().casefold()


def ensure_backend_available(value: str | None) -> str:
    """Return a canonical backend or fail before loading a backend SDK."""

    backend = canonical_backend(value)
    if not backend:
        raise ValueError("vector_backend_not_configured")
    if backend not in AVAILABLE_BACKENDS:
        raise ValueError("vector_backend_unsupported")
    return backend


def backend_status(value: str | None) -> BackendStatus:
    backend = canonical_backend(value)
    if not backend:
        return {
            "backend": "",
            "available": False,
            "reason": "vector backend is not configured",
            "advisory": None,
        }
    if backend not in AVAILABLE_BACKENDS:
        return {
            "backend": backend,
            "available": False,
            "reason": "unsupported vector backend",
            "advisory": None,
        }
    return {
        "backend": backend,
        "available": True,
        "reason": "available",
        "advisory": None,
    }
