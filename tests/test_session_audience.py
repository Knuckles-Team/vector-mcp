"""The engine request context carries the deployment's configured audience."""

from vector_mcp.session import EngineSession


def test_default_audience_is_epistemic_graph() -> None:
    ctx = EngineSession("a", "t", ("r",), ("graph:read",)).engine_verified_context()
    assert ctx["audience"] == "epistemic-graph"


def test_configured_audience_reaches_the_context() -> None:
    session = EngineSession("a", "t", (), ("graph:read",), audience="agent-services")
    assert session.engine_verified_context()["audience"] == "agent-services"
