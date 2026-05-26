"""MCP tools for collection management operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from pydantic import Field

from vector_mcp.auth import get_client


def register_collection_management_tools(mcp: FastMCP):
    @mcp.tool(tags={"collection_management"})
    async def vector_collection_management(
        action: str = Field(
            description="Action to perform. Must be one of: 'create_collection', 'add_documents', 'delete_collection', 'list_collections'"
        ),
        db_type: str | None = Field(default=None, description="db type"),
        db_path: str | None = Field(default=None, description="db path"),
        host: str | None = Field(default=None, description="host"),
        port: str | None = Field(default=None, description="port"),
        db_name: str | None = Field(default=None, description="db name"),
        username: str | None = Field(default=None, description="username"),
        password: str | None = Field(default=None, description="password"),
        collection_name: str | None = Field(
            default=None, description="collection name"
        ),
        overwrite: bool | None = Field(default=None, description="overwrite"),
        document_directory: Path | str | None = Field(
            default=None, description="document directory"
        ),
        document_paths: Path | str | None = Field(
            default=None, description="document paths"
        ),
        document_contents: list[str] | None = Field(
            default=None, description="document contents"
        ),
        confirm: bool | None = Field(default=None, description="confirm"),
        client=Depends(get_client),
    ) -> dict:
        """Manage collection management operations.

        Actions:
          - 'create_collection': Creates a new collection or retrieves an existing one in the vector database.
          - 'add_documents': Adds documents to an existing collection in the vector database.
          - 'delete_collection': Deletes a collection from the vector database.
          - 'list_collections': Lists all collections in the vector database.
        """
        kwargs: dict[str, Any]
        if action == "create_collection":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
                "collection_name": collection_name,
                "overwrite": overwrite,
                "document_directory": document_directory,
                "document_paths": document_paths,
                "document_contents": document_contents,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.create_collection(**kwargs)
        if action == "add_documents":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
                "collection_name": collection_name,
                "document_directory": document_directory,
                "document_paths": document_paths,
                "document_contents": document_contents,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.add_documents(**kwargs)
        if action == "delete_collection":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
                "collection_name": collection_name,
                "confirm": confirm,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.delete_collection(**kwargs)
        if action == "list_collections":
            kwargs = {
                "db_type": db_type,
                "db_path": db_path,
                "host": host,
                "port": port,
                "db_name": db_name,
                "username": username,
                "password": password,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return client.list_collections(**kwargs)
        raise ValueError(
            f"Unknown action: {action}. Must be one of: create_collection', 'add_documents', 'delete_collection', 'list_collections"
        )
