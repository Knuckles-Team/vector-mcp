"""Governed vector retrieval provider package."""

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path


def _source_version() -> str:
    """Read the project version when importing directly from an unpackaged checkout."""
    try:
        project = Path(__file__).resolve().parents[1] / "pyproject.toml"
        return str(
            tomllib.loads(project.read_text(encoding="utf-8"))["project"]["version"]
        )
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return "0.0.0"


try:
    __version__ = _distribution_version("vector-mcp")
except PackageNotFoundError:
    __version__ = _source_version()

from agent_connector_sdk.connector import Connector  # noqa: E402

from vector_mcp.mcp_server import (  # noqa: E402
    register_collection_management_tools,
    register_search_tools,
)

connector = Connector(
    instructions="vector-mcp MCP Server — governed vector retrieval, one tool per operation.",
    action_tools=(
        (
            "collection_management",
            "COLLECTION_MANAGEMENTTOOL",
            register_collection_management_tools,
        ),
        ("search", "SEARCHTOOL", register_search_tools),
    ),
    action_prefixes={
        "vector_collection_management": "vector",
        "vector_search": "vector",
    },
)

__all__ = ["__version__", "connector"]
