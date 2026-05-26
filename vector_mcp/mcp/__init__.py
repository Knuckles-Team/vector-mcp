"""MCP tool registration modules for vector-mcp.

Auto-generated during ecosystem standardization.
Each domain has its own module with a register_*_tools function.
"""

from vector_mcp.mcp.mcp_collection_management import (
    register_collection_management_tools,
)
from vector_mcp.mcp.mcp_search import register_search_tools

__all__ = [
    "register_collection_management_tools",
    "register_search_tools",
]
