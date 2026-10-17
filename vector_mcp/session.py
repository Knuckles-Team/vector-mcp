"""Caller authority and engine connection for the epistemic-graph backends.

The caller's verified identity comes from ``agent_connector_sdk.identity``; the
engine endpoint and its secret come from ``agent_connector_sdk`` engine settings
(credentials by reference only). Nothing here synthesizes authority: without a
bound, authenticated actor that has a tenant, every operation is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_connector_sdk.config import setting
from agent_connector_sdk.credentials.resolution import resolve_secret_reference
from agent_connector_sdk.identity import IdentityRequiredError, current_actor
from agent_connector_sdk.ingest.engine import EngineSettings

__all__ = [
    "EngineNotConfiguredError",
    "EngineSession",
    "engine_connection",
    "resolve_session",
]

_DEFAULT_SCOPES = ("graph:read", "graph:write")
_DEFAULT_AUDIENCE = "epistemic-graph"


class EngineNotConfiguredError(RuntimeError):
    """epistemic-graph is not configured; no local store stands in for it."""


@dataclass(frozen=True)
class EngineSession:
    """The verified authority one engine operation runs under."""

    actor_id: str
    tenant: str
    roles: tuple[str, ...]
    scopes: tuple[str, ...]
    graph: str = "__commons__"
    policy_version: str = "policy:current"

    def engine_verified_context(self) -> dict[str, Any]:
        """The request-context claims the engine verifies."""
        return {
            "principal": self.actor_id,
            "tenant": self.tenant,
            "audience": _DEFAULT_AUDIENCE,
            "agent_id": self.actor_id,
            "roles": sorted(self.roles),
            "scopes": sorted(self.scopes),
            "delegation": [],
            "policy_version": self.policy_version,
        }


def resolve_session(
    session: EngineSession | None = None, *, required_scope: str | None = None
) -> EngineSession:
    """Return the bound caller's authority.

    Raises:
        PermissionError: no authenticated actor with a tenant is bound.
    """
    if session is not None:
        return session
    try:
        actor = current_actor()
    except IdentityRequiredError as exc:
        raise PermissionError(
            "a verified actor is required for this vector operation"
        ) from exc
    if not actor.authenticated or not actor.tenant_id:
        raise PermissionError(
            "vector operations require an authenticated actor with a tenant"
        )
    scopes = {
        item.strip()
        for item in str(
            setting("VECTOR_ENGINE_SCOPES", ",".join(_DEFAULT_SCOPES))
        ).split(",")
        if item.strip()
    }
    if required_scope:
        scopes.add(required_scope)
    engine = EngineSettings.from_settings()
    return EngineSession(
        actor_id=actor.actor_id,
        tenant=actor.tenant_id,
        roles=tuple(actor.roles),
        scopes=tuple(sorted(scopes)),
        graph=engine.graph if engine else "__commons__",
        policy_version=engine.policy_version if engine else "policy:current",
    )


def engine_connection() -> dict[str, Any]:
    """``SyncEpistemicGraphClient.connect`` keywords naming the engine and its secret.

    Raises:
        EngineNotConfiguredError: ``EPISTEMIC_GRAPH_ENDPOINT`` is not set.
    """
    engine = EngineSettings.from_settings()
    if engine is None:
        raise EngineNotConfiguredError(
            "EPISTEMIC_GRAPH_ENDPOINT is not configured; vector storage is "
            "served by epistemic-graph"
        )
    return {
        "auth_secret": resolve_secret_reference(engine.auth_secret_ref, None),
        **engine.address(),
    }
