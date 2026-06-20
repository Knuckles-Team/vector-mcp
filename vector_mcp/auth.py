"""Authentication module for vector-mcp."""

from agent_utilities.core.config import setting


def get_client():
    """Get authenticated client for vector-mcp."""
    from vector_mcp.vector_api import Api

    base_url = setting("LLM_BASE_URL", "")
    token = setting("LLM_TOKEN", "") or setting("LLM_API_KEY", "")
    verify = setting("LLM_SSL_VERIFY", False)
    if not base_url:
        raise RuntimeError("LLM_BASE_URL not set")
    return Api(base_url=base_url, token=token, verify=verify)
