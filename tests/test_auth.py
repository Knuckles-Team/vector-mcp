import os
from unittest.mock import patch
import pytest

from agent_utilities.base_utilities import to_boolean
from vector_mcp.auth import get_client


@pytest.fixture
def mock_api_class():
    """Mock the Api class imported by auth."""
    with patch("vector_mcp.vector_api.Api") as mock:
        yield mock


def test_get_client_missing_base_url():
    """get_client should raise RuntimeError if LLM_BASE_URL is not set."""
    with patch.dict(os.environ, {}, clear=True):
        if "LLM_BASE_URL" in os.environ:
            del os.environ["LLM_BASE_URL"]
        with pytest.raises(RuntimeError, match="LLM_BASE_URL not set"):
            get_client()


def test_get_client_success(mock_api_class):
    """get_client should correctly parse env variables and return the Api client."""
    with patch.dict(
        os.environ,
        {
            "LLM_BASE_URL": "http://test-url",
            "LLM_TOKEN": "test-token",
            "LLM_SSL_VERIFY": "True",
        },
    ):
        client = get_client()
        assert client is not None
        mock_api_class.assert_called_once_with(
            base_url="http://test-url", token="test-token", verify=True
        )


def test_get_client_verify_defaults(mock_api_class):
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
        client = get_client()
        assert client is not None
        mock_api_class.assert_called_once_with(
            base_url="http://test-url", token="test-key", verify=False
        )
