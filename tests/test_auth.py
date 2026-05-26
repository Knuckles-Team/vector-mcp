import os
import sys
from unittest.mock import MagicMock, patch
import pytest

from agent_utilities.base_utilities import to_boolean

# Share the same vector_api mock across all tests to prevent import-caching issues
if "vector_mcp.vector_api" not in sys.modules:
    mock_vector_api = MagicMock()
    mock_vector_api.Api = MagicMock()
    sys.modules["vector_mcp.vector_api"] = mock_vector_api
else:
    mock_vector_api = sys.modules["vector_mcp.vector_api"]

MockApiClass = mock_vector_api.Api

from vector_mcp.auth import get_client


def test_get_client_missing_base_url():
    """get_client should raise RuntimeError if LLM_BASE_URL is not set."""
    with patch.dict(os.environ, {}, clear=True):
        if "LLM_BASE_URL" in os.environ:
            del os.environ["LLM_BASE_URL"]
        with pytest.raises(RuntimeError, match="LLM_BASE_URL not set"):
            get_client()


def test_get_client_success():
    """get_client should correctly parse env variables and return the Api client."""
    with patch.dict(
        os.environ,
        {
            "LLM_BASE_URL": "http://test-url",
            "LLM_TOKEN": "test-token",
            "LLM_SSL_VERIFY": "True",
        },
    ):
        MockApiClass.reset_mock()
        client = get_client()
        assert client is not None
        MockApiClass.assert_called_once_with(
            base_url="http://test-url", token="test-token", verify=True
        )


def test_get_client_verify_defaults():
    """get_client should default verify to False and check fallback token env variables."""
    with patch.dict(
        os.environ,
        {
            "LLM_BASE_URL": "http://test-url",
            "LLM_API_KEY": "test-key",
        },
    ):
        if "LLM_TOKEN" in os.environ:
            del os.environ["LLM_TOKEN"]
        MockApiClass.reset_mock()
        client = get_client()
        assert client is not None
        MockApiClass.assert_called_once_with(
            base_url="http://test-url", token="test-key", verify=False
        )
