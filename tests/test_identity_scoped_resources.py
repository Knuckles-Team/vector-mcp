"""Identity-scoped collection auto-load (CONCEPT:AU-OS.identity.identity-scoped-resource-autoload).

``vector_collection_management(action="list_collections")`` routes the
returned collection names through the shared agent-utilities entitlement
resolver so a caller's Okta/Keycloak groups scope which collections they
see. The resolver itself is tested in agent-utilities; here we mock the
entitlement source (per the container-manager-mcp reference pattern) and
test the local filtering logic plus its fail-open degrade when
agent-utilities predates the resolver.
"""

import os
import sys
import types

import pytest

os.environ.setdefault("LLM_BASE_URL", "http://test-url")

# Mock the missing vector_api module so that get_client works, mirroring
# tests/test_mcp_server.py's import-time setup.
if "vector_mcp.vector_api" not in sys.modules:
    from unittest.mock import MagicMock

    mock_vector_api = MagicMock()
    mock_vector_api.Api = MagicMock()
    sys.modules["vector_mcp.vector_api"] = mock_vector_api

from vector_mcp.mcp_server import _entitled


def _install_fake_entitlements(entitled_names):
    """Install a fake ``agent_utilities.security.entitlements`` module."""
    module = types.ModuleType("agent_utilities.security.entitlements")
    module.identity_scoped_resources = lambda namespace, names: [
        n for n in names if n in entitled_names
    ]
    sys.modules["agent_utilities.security.entitlements"] = module


@pytest.fixture(autouse=True)
def _cleanup_fake_module():
    yield
    sys.modules.pop("agent_utilities.security.entitlements", None)


class TestEntitledHelper:
    def test_filters_to_entitled_subset(self):
        _install_fake_entitlements({"docs"})
        assert _entitled("collection", ["docs", "logs", "scratch"]) == ["docs"]

    def test_no_entitlements_returns_empty(self):
        _install_fake_entitlements(set())
        assert _entitled("collection", ["docs", "logs"]) == []

    def test_degrades_to_full_set_without_resolver(self):
        # agent-utilities predates the resolver (ImportError) -> fail open to
        # today's behaviour, not fail closed.
        sys.modules.pop("agent_utilities.security.entitlements", None)
        assert _entitled("collection", ["docs", "logs"]) == ["docs", "logs"]
