"""Shared test fixtures for Vector MCP."""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Select the native provider without adding endpoint configuration."""
    monkeypatch.setenv("DATABASE_TYPE", "epistemic_graph")
