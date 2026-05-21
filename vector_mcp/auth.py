"""Authentication module for vector-mcp."""

import os

from agent_utilities.base_utilities import to_boolean


def get_client():
    """Get authenticated client for vector-mcp."""
    from vector_mcp.vector_api import Api

    base_url = os.getenv("LLM_BASE_URL")
    token = os.getenv("LLM_TOKEN", os.getenv("LLM_API_KEY"))
    verify = to_boolean(os.getenv("LLM_SSL_VERIFY", "False"))
    if not base_url:
        raise RuntimeError("LLM_BASE_URL not set")
    return Api(base_url=base_url, token=token, verify=verify)
