from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import agent_utilities

agent_server_module = importlib.import_module("vector_mcp.agent_server")


def _arguments(**overrides):
    values = {
        "debug": False,
        "mcp_url": None,
        "mcp_config": None,
        "host": "127.0.0.1",
        "port": 8888,
        "provider": "configured-provider",
        "model_id": "configured-model",
        "base_url": "https://model.example.invalid/v1",
        "api_key": None,
        "custom_skills_directory": None,
        "workspace": None,
        "web": False,
        "terminal": False,
        "web_logs": False,
        "otel": False,
        "otel_endpoint": None,
        "otel_headers": None,
        "otel_public_key": None,
        "otel_secret_key": None,
        "otel_protocol": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _configure_agent_utilities(monkeypatch, *, args, identity):
    parser = MagicMock()
    parser.parse_args.return_value = args
    create_server = MagicMock()
    initialize_workspace = MagicMock()
    build_prompt = MagicMock(return_value="generated prompt")

    monkeypatch.setattr(agent_utilities, "create_agent_parser", lambda: parser)
    monkeypatch.setattr(agent_utilities, "create_agent_server", create_server)
    monkeypatch.setattr(agent_utilities, "initialize_workspace", initialize_workspace)
    monkeypatch.setattr(agent_utilities, "load_identity", lambda: identity)
    monkeypatch.setattr(
        agent_utilities, "build_system_prompt_from_workspace", build_prompt
    )
    monkeypatch.setattr(
        agent_server_module,
        "setting",
        lambda _name, default=None: default,
    )
    return create_server, initialize_workspace, build_prompt


def test_agent_server_uses_explicit_operator_config(monkeypatch) -> None:
    args = _arguments(mcp_config="operator-config.json", debug=True)
    create_server, initialize_workspace, _ = _configure_agent_utilities(
        monkeypatch,
        args=args,
        identity={
            "name": "Vector MCP Test",
            "description": "Test description",
            "content": "Test prompt",
        },
    )

    agent_server_module.agent_server()

    initialize_workspace.assert_called_once()
    create_server.assert_called_once_with(
        mcp_url=None,
        mcp_config="operator-config.json",
        host="127.0.0.1",
        port=8888,
        provider="configured-provider",
        model_id="configured-model",
        router_model="configured-model",
        agent_model="configured-model",
        base_url="https://model.example.invalid/v1",
        api_key=None,
        custom_skills_directory=None,
        enable_web_ui=False,
        enable_terminal_ui=False,
        enable_web_logs=False,
        workspace=None,
        name="Vector MCP Test",
        system_prompt="Test prompt",
        enable_otel=False,
        otel_endpoint=None,
        otel_headers=None,
        otel_public_key=None,
        otel_secret_key=None,
        otel_protocol=None,
        debug=True,
    )


def test_agent_server_uses_packaged_neutral_config(monkeypatch) -> None:
    args = _arguments()
    create_server, _, build_prompt = _configure_agent_utilities(
        monkeypatch,
        args=args,
        identity={
            "name": "Vector MCP",
            "description": "Fallback description",
            "content": "",
        },
    )

    agent_server_module.agent_server()

    build_prompt.assert_called_once()
    configured_path = create_server.call_args.kwargs["mcp_config"]
    assert configured_path.endswith("bundled_mcp.json")
