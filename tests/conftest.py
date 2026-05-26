"""Shared test fixtures for Vector Mcp."""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("VECTOR_URL", "https://test.example.com")
    monkeypatch.setenv("VECTOR_TOKEN", "test-token-12345")
    monkeypatch.setenv("VECTOR_SSL_VERIFY", "False")
