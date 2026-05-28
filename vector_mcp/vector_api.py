"""API client stub for vector-mcp."""

from typing import Any


class Api:
    """API Client for vector-mcp."""

    def __init__(self, base_url: str, token: str | None = None, verify: bool = False):
        self.base_url = base_url
        self.token = token
        self.verify = verify

    def create_collection(self, *args: Any, **kwargs: Any) -> dict:
        """Create a collection."""
        return {}

    def add_documents(self, *args: Any, **kwargs: Any) -> dict:
        """Add documents."""
        return {}

    def delete_collection(self, *args: Any, **kwargs: Any) -> dict:
        """Delete a collection."""
        return {}

    def list_collections(self, *args: Any, **kwargs: Any) -> dict:
        """List collections."""
        return {}

    def semantic_search(self, *args: Any, **kwargs: Any) -> dict:
        """Perform semantic search."""
        return {}

    def lexical_search(self, *args: Any, **kwargs: Any) -> dict:
        """Perform lexical search."""
        return {}

    def search(self, *args: Any, **kwargs: Any) -> dict:
        """Perform hybrid search."""
        return {}
