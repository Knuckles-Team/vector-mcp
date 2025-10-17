#!/usr/bin/python
# coding: utf-8
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Union
from pydantic import Field
from fastmcp import FastMCP, Context
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
from fastmcp.server.auth.providers.jwt import JWTVerifier, StaticTokenVerifier
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.retriever.pgvector_retriever import PGVectorRetriever
from vector_mcp.retriever.qdrant_retriever import QdrantRetriever
from vector_mcp.retriever.couchbase_retriever import CouchbaseRetriever
from vector_mcp.retriever.mongodb_retriever import MongoDBRetriever
from vector_mcp.retriever.chromadb_retriever import ChromaDBRetriever
from vector_mcp.vectordb.utils import get_logger

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger("VectorServer")


environment_db_type = os.environ.get("DATABASE_TYPE", "chromadb").lower()
environment_db_path = os.environ.get("DATABASE_PATH", os.path.expanduser("~"))
environment_host = os.environ.get("HOST", None)
environment_port = os.environ.get("PORT", None)
environment_db_name = os.environ.get("DBNAME", "memory")
environment_username = os.environ.get("USERNAME", None)
environment_password = os.environ.get("PASSWORD", None)
environment_api_token = os.environ.get("API_TOKEN", None)
environment_collection_name = os.environ.get("COLLECTION_NAME", "memory")
environment_document_directory = os.environ.get("DOCUMENT_DIRECTORY", None)


mcp = FastMCP(name="VectorServer")
mcp.on_duplicate_tools = "error"


def initialize_retriever(
    db_type: str = Field(
        description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)",
        default="chromadb",
    ),
    db_path: str = Field(
        description="The path to store chromadb files",
        default=environment_db_path,
    ),
    host: Optional[str] = Field(
        description="Hostname or IP address of the database server",
        default=environment_host,
    ),
    port: Optional[str] = Field(
        description="Port number of the database server", default=environment_port
    ),
    db_name: Optional[str] = Field(
        description="Name of the database or path (depending on DB type)",
        default=environment_db_name,
    ),
    username: Optional[str] = Field(
        description="Username for database authentication", default=environment_username
    ),
    password: Optional[str] = Field(
        description="Password for database authentication", default=environment_password
    ),
    api_token: Optional[str] = Field(
        description="API Token for database authentication",
        default=environment_api_token,
    ),
    collection_name: str = Field(
        description="The name of the collection to initialize the database with",
        default=environment_collection_name,
    ),
) -> RAGRetriever:
    try:
        db_type_lower = db_type.strip().lower()
        if db_type_lower == "chromadb":
            if host and port:
                retriever: RAGRetriever = ChromaDBRetriever(
                    host=host, port=int(port), collection_name=collection_name
                )
            else:
                retriever: RAGRetriever = ChromaDBRetriever(
                    path=os.path.join(db_path, db_name), collection_name=collection_name
                )
        elif db_type_lower == "pgvector":
            retriever: RAGRetriever = PGVectorRetriever(
                host=host,
                port=port,
                dbname=db_name,
                username=username,
                password=password,
                collection_name=collection_name,
            )
        elif db_type_lower == "qdrant":
            client_kwargs = {}
            if host:
                client_kwargs = {"host": host} if host else {"location": ":memory:"}
            if port:
                client_kwargs["port"] = str(port)
            if password:
                client_kwargs["api_key"] = api_token
            retriever: RAGRetriever = QdrantRetriever(
                client_kwargs=client_kwargs, collection_name=collection_name
            )
        elif db_type_lower == "couchbase":
            connection_string = (
                f"couchbase://{host}" if host else "couchbase://localhost"
            )
            if port:
                connection_string += f":{port}"
            retriever: RAGRetriever = CouchbaseRetriever(
                connection_string=connection_string,
                username=username,
                password=password,
                bucket_name=db_name,
                collection_name=collection_name,
            )
        elif db_type_lower == "mongodb":
            connection_string = ""
            if host:
                connection_string = (
                    f"mongodb://{username}:{password}@{host}:{port or '27017'}/{db_name}"
                    if username and password
                    else f"mongodb://{host}:{port or '27017'}/{db_name}"
                )
            retriever: RAGRetriever = MongoDBRetriever(
                connection_string=connection_string,
                database_name=db_name,
                collection_name=collection_name,
            )
        else:
            logger.error("Failed to identify vector database from supported databases")
            sys.exit(1)
        logger.info("Vector Database initialized successfully.")
        retriever.connect_database(collection_name=collection_name)
        return retriever
    except Exception as e:
        logger.error(f"Failed to initialize Vector Database: {str(e)}")
        raise e


@mcp.tool(
    annotations={
        "title": "Create a Collection",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"collection_management"},
)
async def create_collection(
    db_type: str = Field(
        description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)",
        default=environment_db_type,
    ),
    db_path: str = Field(
        description="The path to store chromadb files",
        default=environment_db_path,
    ),
    host: Optional[str] = Field(
        description="Hostname or IP address of the database server",
        default=environment_host,
    ),
    port: Optional[str] = Field(
        description="Port number of the database server", default=environment_port
    ),
    db_name: Optional[str] = Field(
        description="Name of the database or path (depending on DB type)",
        default=environment_db_name,
    ),
    username: Optional[str] = Field(
        description="Username for database authentication", default=environment_username
    ),
    password: Optional[str] = Field(
        description="Password for database authentication", default=environment_password
    ),
    collection_name: str = Field(
        description="Name of the collection to create or retrieve",
        default=environment_collection_name,
    ),
    overwrite: Optional[bool] = Field(
        description="Whether to overwrite the collection if it exists", default=False
    ),
    document_directory: Optional[Union[Path, str]] = Field(
        description="Document directory to read documents from",
        default=environment_document_directory,
    ),
    document_paths: Optional[Union[Path, str]] = Field(
        description="Document paths on the file system or URLs to read from",
        default=None,
    ),
    ctx: Context = Field(
        description="FastMCP context for progress reporting", default=None
    ),
) -> Dict:
    """Creates a new collection or retrieves an existing one in the vector database."""
    if not collection_name:
        raise ValueError("collection_name must not be empty")

    retriever = initialize_retriever(
        db_type=db_type,
        db_path=db_path,
        host=host,
        port=port,
        db_name=db_name,
        username=username,
        password=password,
        collection_name=collection_name,
    )

    logger.debug(
        f"Creating collection: {collection_name}, overwrite: {overwrite},\n"
        f"document directory: {document_directory}, document urls: {document_paths}"
    )
    response = {
        "message": "Collection created or retrieved successfully.",
        "data": {
            "Database Type": db_type,
            "Collection Name": collection_name,
            "Overwrite": overwrite,
            "Document Directory": document_directory,
            "Document Paths": document_paths,
            "Database": db_name,
            "Database Host": host,
        },
        "status": 200,
    }
    try:
        if ctx:
            await ctx.report_progress(progress=0, total=100)
        coll = retriever.initialize_collection(
            collection_name=collection_name,
            overwrite=overwrite,
            document_directory=document_directory,
            document_paths=document_paths,
        )
        if ctx:
            await ctx.report_progress(progress=100, total=100)
        else:
            response["message"] = "Collection failed to be created."
            response["status"] = 403
        response["completion"] = coll
        return response
    except ValueError as e:
        logger.error(f"Invalid input for create_collection: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        raise RuntimeError(f"Failed to create collection: {str(e)}")


@mcp.tool(
    annotations={
        "title": "Retrieve Texts from a Collection",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"retrieve"},
)
async def retrieve(
    db_type: str = Field(
        description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)",
        default=environment_db_type,
    ),
    db_path: str = Field(
        description="The path to store chromadb files",
        default=environment_db_path,
    ),
    host: Optional[str] = Field(
        description="Hostname or IP address of the database server",
        default=environment_host,
    ),
    port: Optional[str] = Field(
        description="Port number of the database server", default=environment_port
    ),
    db_name: Optional[str] = Field(
        description="Name of the database or path (depending on DB type)",
        default=environment_db_name,
    ),
    username: Optional[str] = Field(
        description="Username for database authentication", default=environment_username
    ),
    password: Optional[str] = Field(
        description="Password for database authentication", default=environment_password
    ),
    collection_name: str = Field(
        description="Name of the collection to retrieve",
        default=environment_collection_name,
    ),
    question: str = Field(
        description="The question or phrase to similarity search in the vector database",
        default=None,
    ),
    number_results: int = Field(
        description="The total number of retrieved document texts to provide", default=1
    ),
    ctx: Context = Field(
        description="FastMCP context for progress reporting", default=None
    ),
) -> Dict:
    """Retrieves and gathers related knowledge from the vector database instance using the question variable.
    This can be used as a primary source of knowledge retrieval.
    It will return relevant text(s) which should be parsed for the most
    relevant information pertaining to the question and summarized as the final output
    """
    logger.debug(f"Initializing collection: {collection_name}")

    retriever = initialize_retriever(
        db_type=db_type,
        db_path=db_path,
        host=host,
        port=port,
        db_name=db_name,
        username=username,
        password=password,
        collection_name=collection_name,
    )

    try:
        if ctx:
            await ctx.report_progress(progress=0, total=100)
        logger.debug(f"Querying collection: {question}")
        texts = retriever.query(question=question, number_results=number_results)
        if ctx:
            await ctx.report_progress(progress=100, total=100)
        response = {
            "retrieved_texts": texts,
            "message": "Collection retrieved from successfully",
            "data": {
                "Database Type": db_type,
                "Collection Name": collection_name,
                "Question": question,
                "Number of Results": number_results,
                "Database": db_name,
                "Database Host": host,
            },
            "status": 200,
        }
        return response
    except ValueError as e:
        logger.error(f"Invalid input for get_collection: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to get collection: {str(e)}")
        raise RuntimeError(f"Failed to get collection: {str(e)}")


@mcp.tool(
    annotations={
        "title": "Add Documents to a Collection",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"collection_management"},
)
async def add_documents(
    db_type: str = Field(
        description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)",
        default=environment_db_type,
    ),
    db_path: str = Field(
        description="The path to store chromadb files",
        default=environment_db_path,
    ),
    host: Optional[str] = Field(
        description="Hostname or IP address of the database server",
        default=environment_host,
    ),
    port: Optional[str] = Field(
        description="Port number of the database server", default=environment_port
    ),
    db_name: Optional[str] = Field(
        description="Name of the database or path (depending on DB type)",
        default=environment_db_name,
    ),
    username: Optional[str] = Field(
        description="Username for database authentication", default=environment_username
    ),
    password: Optional[str] = Field(
        description="Password for database authentication", default=environment_password
    ),
    collection_name: str = Field(
        description="Name of the target collection.", default=None
    ),
    document_directory: Optional[Union[Path, str]] = Field(
        description="Document directory to read documents from",
        default=environment_document_directory,
    ),
    document_paths: Optional[Union[Path, str]] = Field(
        description="Document paths on the file system or URLs to read from",
        default=None,
    ),
    ctx: Context = Field(
        description="FastMCP context for progress reporting", default=None
    ),
) -> Dict:
    """Adds documents to an existing collection in the vector database.
    This can be used to extend collections with additional documents"""
    if not document_directory and document_paths:
        raise ValueError("docs list must not be empty")

    retriever = initialize_retriever(
        db_type=db_type,
        db_path=db_path,
        host=host,
        port=port,
        db_name=db_name,
        username=username,
        password=password,
        collection_name=collection_name,
    )
    logger.debug(
        f"Inserting {document_paths} documents into collection: {collection_name}, document_directory: {document_directory}"
    )

    try:
        if ctx:
            await ctx.report_progress(progress=0, total=100)
        texts = retriever.add_documents(
            document_directory=document_directory,
            document_paths=document_paths,
        )
        if ctx:
            await ctx.report_progress(progress=100, total=100)

        response = {
            "added_texts": texts,
            "message": "Collection retrieved from successfully",
            "data": {
                "Database Type": db_type,
                "Collection Name": collection_name,
                "Document Directory": document_directory,
                "Document Paths": document_paths,
                "Database": db_name,
                "Database Host": host,
            },
            "status": 200,
        }
        return response
    except ValueError as e:
        logger.error(f"Invalid input for insert_documents: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to insert documents: {str(e)}")
        raise RuntimeError(f"Failed to insert documents: {str(e)}")


@mcp.tool(
    annotations={
        "title": "Delete a Collection",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"collection_management"},
)
async def delete_collection(
    db_type: str = Field(
        description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)",
        default=environment_db_type,
    ),
    db_path: str = Field(
        description="The path to store chromadb files",
        default=environment_db_path,
    ),
    host: Optional[str] = Field(
        description="Hostname or IP address of the database server",
        default=environment_host,
    ),
    port: Optional[str] = Field(
        description="Port number of the database server", default=environment_port
    ),
    db_name: Optional[str] = Field(
        description="Name of the database or path (depending on DB type)",
        default=environment_db_name,
    ),
    username: Optional[str] = Field(
        description="Username for database authentication", default=environment_username
    ),
    password: Optional[str] = Field(
        description="Password for database authentication", default=environment_password
    ),
    collection_name: str = Field(
        description="Name of the target collection.", default=None
    ),
    ctx: Context = Field(
        description="FastMCP context for progress reporting", default=None
    ),
) -> Dict:
    """Deletes a collection from the vector database."""

    retriever = initialize_retriever(
        db_type=db_type,
        db_path=db_path,
        host=host,
        port=port,
        db_name=db_name,
        username=username,
        password=password,
        collection_name=collection_name,
    )
    logger.debug(f"Deleting collection: {collection_name} from: {db_type}")

    try:
        if ctx:
            await ctx.report_progress(progress=0, total=100)
        retriever.vector_db.delete_collection(collection_name=collection_name)
        if ctx:
            await ctx.report_progress(progress=100, total=100)
        response = {
            "message": f"Collection {collection_name} deleted successfully",
            "data": {
                "Database Type": db_type,
                "Collection Name": collection_name,
                "Database": db_name,
                "Database Host": host,
            },
            "status": 200,
        }
        return response
    except ValueError as e:
        logger.error(f"Invalid input for delete collection: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection: {str(e)}")
        raise RuntimeError(f"Failed to delete collection: {str(e)}")


def vector_mcp():
    parser = argparse.ArgumentParser(
        description="Create, manage, and retrieve from collections in a vector database"
    )
    parser.add_argument(
        "-t",
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="Transport method: 'stdio', 'http', or 'sse' [legacy] (default: stdio)",
    )
    parser.add_argument(
        "-s",
        "--host",
        default="0.0.0.0",
        help="Host address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port number for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--auth-type",
        default="none",
        choices=["none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"],
        help="Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none)",
    )
    # JWT/Token params
    parser.add_argument(
        "--token-jwks-uri", default=None, help="JWKS URI for JWT verification"
    )
    parser.add_argument(
        "--token-issuer", default=None, help="Issuer for JWT verification"
    )
    parser.add_argument(
        "--token-audience", default=None, help="Audience for JWT verification"
    )
    # OAuth Proxy params
    parser.add_argument(
        "--oauth-upstream-auth-endpoint",
        default=None,
        help="Upstream authorization endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-token-endpoint",
        default=None,
        help="Upstream token endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-id",
        default=None,
        help="Upstream client ID for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-secret",
        default=None,
        help="Upstream client secret for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-base-url", default=None, help="Base URL for OAuth Proxy"
    )
    # OIDC Proxy params
    parser.add_argument(
        "--oidc-config-url", default=None, help="OIDC configuration URL"
    )
    parser.add_argument("--oidc-client-id", default=None, help="OIDC client ID")
    parser.add_argument("--oidc-client-secret", default=None, help="OIDC client secret")
    parser.add_argument("--oidc-base-url", default=None, help="Base URL for OIDC Proxy")
    # Remote OAuth params
    parser.add_argument(
        "--remote-auth-servers",
        default=None,
        help="Comma-separated list of authorization servers for Remote OAuth",
    )
    parser.add_argument(
        "--remote-base-url", default=None, help="Base URL for Remote OAuth"
    )
    # Common
    parser.add_argument(
        "--allowed-client-redirect-uris",
        default=None,
        help="Comma-separated list of allowed client redirect URIs",
    )
    # Eunomia params
    parser.add_argument(
        "--eunomia-type",
        default="none",
        choices=["none", "embedded", "remote"],
        help="Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none)",
    )
    parser.add_argument(
        "--eunomia-policy-file",
        default="mcp_policies.json",
        help="Policy file for embedded Eunomia (default: mcp_policies.json)",
    )
    parser.add_argument(
        "--eunomia-remote-url", default=None, help="URL for remote Eunomia server"
    )

    args = parser.parse_args()

    if args.port < 0 or args.port > 65535:
        print(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)

    # Set auth based on type
    auth = None
    allowed_uris = (
        args.allowed_client_redirect_uris.split(",")
        if args.allowed_client_redirect_uris
        else None
    )

    if args.auth_type == "none":
        auth = None
    elif args.auth_type == "static":
        # Internal static tokens (hardcoded example)
        auth = StaticTokenVerifier(
            tokens={
                "test-token": {"client_id": "test-user", "scopes": ["read", "write"]},
                "admin-token": {"client_id": "admin", "scopes": ["admin"]},
            }
        )
    elif args.auth_type == "jwt":
        if not (args.token_jwks_uri and args.token_issuer and args.token_audience):
            print(
                "Error: jwt requires --token-jwks-uri, --token-issuer, --token-audience"
            )
            sys.exit(1)
        auth = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
    elif args.auth_type == "oauth-proxy":
        if not (
            args.oauth_upstream_auth_endpoint
            and args.oauth_upstream_token_endpoint
            and args.oauth_upstream_client_id
            and args.oauth_upstream_client_secret
            and args.oauth_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            print(
                "Error: oauth-proxy requires --oauth-upstream-auth-endpoint, --oauth-upstream-token-endpoint, --oauth-upstream-client-id, --oauth-upstream-client-secret, --oauth-base-url, --token-jwks-uri, --token-issuer, --token-audience"
            )
            sys.exit(1)
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = OAuthProxy(
            upstream_authorization_endpoint=args.oauth_upstream_auth_endpoint,
            upstream_token_endpoint=args.oauth_upstream_token_endpoint,
            upstream_client_id=args.oauth_upstream_client_id,
            upstream_client_secret=args.oauth_upstream_client_secret,
            token_verifier=token_verifier,
            base_url=args.oauth_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "oidc-proxy":
        if not (
            args.oidc_config_url
            and args.oidc_client_id
            and args.oidc_client_secret
            and args.oidc_base_url
        ):
            print(
                "Error: oidc-proxy requires --oidc-config-url, --oidc-client-id, --oidc-client-secret, --oidc-base-url"
            )
            sys.exit(1)
        auth = OIDCProxy(
            config_url=args.oidc_config_url,
            client_id=args.oidc_client_id,
            client_secret=args.oidc_client_secret,
            base_url=args.oidc_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "remote-oauth":
        if not (
            args.remote_auth_servers
            and args.remote_base_url
            and args.token_jwks_uri
            and args.token_issuer
            and args.token_audience
        ):
            print(
                "Error: remote-oauth requires --remote-auth-servers, --remote-base-url, --token-jwks-uri, --token-issuer, --token-audience"
            )
            sys.exit(1)
        auth_servers = [url.strip() for url in args.remote_auth_servers.split(",")]
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        auth = RemoteAuthProvider(
            token_verifier=token_verifier,
            authorization_servers=auth_servers,
            base_url=args.remote_base_url,
        )
    mcp.auth = auth
    if args.eunomia_type != "none":
        from eunomia_mcp import create_eunomia_middleware

        if args.eunomia_type == "embedded":
            if not args.eunomia_policy_file:
                print("Error: embedded Eunomia requires --eunomia-policy-file")
                sys.exit(1)
            middleware = create_eunomia_middleware(policy_file=args.eunomia_policy_file)
            mcp.add_middleware(middleware)
        elif args.eunomia_type == "remote":
            if not args.eunomia_remote_url:
                print("Error: remote Eunomia requires --eunomia-remote-url")
                sys.exit(1)
            middleware = create_eunomia_middleware(
                use_remote_eunomia=args.eunomia_remote_url
            )
            mcp.add_middleware(middleware)

    mcp.add_middleware(
        ErrorHandlingMiddleware(include_traceback=True, transform_errors=True)
    )
    mcp.add_middleware(
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20)
    )
    mcp.add_middleware(TimingMiddleware())
    mcp.add_middleware(LoggingMiddleware())

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger = logging.getLogger("Vector")
        logger.error("Transport not supported")
        sys.exit(1)


if __name__ == "__main__":
    vector_mcp()
