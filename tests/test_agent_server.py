import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock the agent_utilities module before importing agent_server
mock_agent_utilities = MagicMock()
sys.modules["agent_utilities"] = mock_agent_utilities

from vector_mcp.agent_server import agent_server


def test_agent_server_success():
    """Verify that agent_server CLI parses args and invokes create_agent_server correctly."""
    mock_agent_utilities.reset_mock()

    mock_agent_utilities.load_identity.return_value = {
        "name": "Vector Mcp Test",
        "description": "Test description",
        "content": "Test prompt",
    }

    mock_parser = MagicMock()
    mock_args = MagicMock()
    mock_args.debug = True
    mock_args.mcp_url = "http://test-mcp"
    mock_args.mcp_config = "test_config.json"
    mock_args.host = "127.0.0.1"
    mock_args.port = 8888
    mock_args.provider = "test-provider"
    mock_args.model_id = "test-model"
    mock_args.base_url = "http://base-url"
    mock_args.api_key = "test-key"
    mock_args.custom_skills_directory = "/custom"
    mock_args.web = True
    mock_args.otel = True
    mock_args.otel_endpoint = "http://otel"
    mock_args.otel_headers = "headers"
    mock_args.otel_public_key = "pub"
    mock_args.otel_secret_key = "sec"
    mock_args.otel_protocol = "grpc"

    mock_parser.parse_args.return_value = mock_args
    mock_agent_utilities.create_agent_parser.return_value = mock_parser

    with patch("sys.argv", ["agent_server"]), patch("builtins.print") as mock_print:
        agent_server()

        mock_agent_utilities.initialize_workspace.assert_called_once()
        mock_agent_utilities.load_identity.assert_called_once()
        mock_agent_utilities.create_agent_parser.assert_called_once()
        mock_agent_utilities.create_agent_server.assert_called_once_with(
            mcp_url="http://test-mcp",
            mcp_config="test_config.json",
            host="127.0.0.1",
            port=8888,
            provider="test-provider",
            model_id="test-model",
            router_model="test-model",
            agent_model="test-model",
            base_url="http://base-url",
            api_key="test-key",
            custom_skills_directory="/custom",
            enable_web_ui=True,
            enable_otel=True,
            otel_endpoint="http://otel",
            otel_headers="headers",
            otel_public_key="pub",
            otel_secret_key="sec",
            otel_protocol="grpc",
            debug=True,
        )


def test_agent_server_fallback_system_prompt():
    """Verify system prompt generation fallback when identity content is empty."""
    mock_agent_utilities.reset_mock()

    mock_agent_utilities.load_identity.return_value = {
        "name": "Vector Mcp Fallback",
        "description": "Fallback description",
        "content": "",
    }
    mock_agent_utilities.build_system_prompt_from_workspace.return_value = (
        "generated-prompt"
    )

    mock_parser = MagicMock()
    mock_args = MagicMock()
    mock_args.debug = False
    mock_args.mcp_url = None
    mock_args.mcp_config = None
    mock_args.host = None
    mock_args.port = None
    mock_args.provider = None
    mock_args.model_id = None
    mock_args.base_url = None
    mock_args.api_key = None
    mock_args.custom_skills_directory = None
    mock_args.web = False
    mock_args.otel = False
    mock_args.otel_endpoint = None
    mock_args.otel_headers = None
    mock_args.otel_public_key = None
    mock_args.otel_secret_key = None
    mock_args.otel_protocol = None

    mock_parser.parse_args.return_value = mock_args
    mock_agent_utilities.create_agent_parser.return_value = mock_parser

    with patch("sys.argv", ["agent_server"]), patch("builtins.print") as mock_print:
        agent_server()

        mock_agent_utilities.build_system_prompt_from_workspace.assert_called_once()
        mock_agent_utilities.create_agent_server.assert_called_once()
