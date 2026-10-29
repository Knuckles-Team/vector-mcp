"""The ChromaDB data directory is owned by this connector, not agent-utilities."""

import importlib

import pytest

pytest.importorskip("platformdirs")


def _module():
    return importlib.import_module("vector_mcp.vectordb.chromadb")


def test_data_dir_defaults_to_the_connector_app_name(monkeypatch, tmp_path):
    monkeypatch.delenv("VECTOR_MCP_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    path = _module()._data_dir()
    assert path == tmp_path / "vector-mcp"
    assert "agent-utilities" not in str(path)


def test_data_dir_honours_connector_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_MCP_DATA_DIR", str(tmp_path / "chosen"))
    assert _module()._data_dir() == tmp_path / "chosen"


def test_legacy_agent_utilities_data_dir_env_is_ignored(monkeypatch, tmp_path):
    monkeypatch.delenv("VECTOR_MCP_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "legacy"))
    assert _module()._data_dir() == tmp_path / "vector-mcp"
