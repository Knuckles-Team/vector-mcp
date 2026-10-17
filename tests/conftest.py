"""Shared test fixtures for Vector MCP."""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Select the native provider without adding endpoint configuration."""
    monkeypatch.setenv("DATABASE_TYPE", "epistemic_graph")


@pytest.fixture(autouse=True)
def _embedding_model_configured(monkeypatch):
    """The SDK embedding client refuses to build without a configured model."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
