#!/usr/bin/python
# coding: utf-8
import os
import argparse
import sys
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Union

import requests
from eunomia_mcp.middleware import EunomiaMcpMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from pydantic import Field
from fastmcp import FastMCP, Context
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
from fastmcp.server.auth.providers.jwt import JWTVerifier, StaticTokenVerifier
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.utilities.logging import get_logger
from vector_mcp.retriever.retriever import RAGRetriever
from vector_mcp.utils import to_integer, to_boolean
from vector_mcp.middlewares import UserTokenMiddleware, JWTClaimsLoggingMiddleware

__version__ = "1.0.7"

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger("VectorServer")

config = {
    "enable_delegation": to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
    "audience": os.environ.get("AUDIENCE", None),
    "delegated_scopes": os.environ.get("DELEGATED_SCOPES", "api"),
    "token_endpoint": None,  # Will be fetched dynamically from OIDC config
    "oidc_client_id": os.environ.get("OIDC_CLIENT_ID", None),
    "oidc_client_secret": os.environ.get("OIDC_CLIENT_SECRET", None),
    "oidc_config_url": os.environ.get("OIDC_CONFIG_URL", None),
    "jwt_jwks_uri": os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI", None),
    "jwt_issuer": os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER", None),
    "jwt_audience": os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE", None),
    "jwt_algorithm": os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM", None),
    "jwt_secret": os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY", None),
    "jwt_required_scopes": os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES", None),
}

DEFAULT_TRANSPORT = os.environ.get("TRANSPORT", "stdio")
DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(os.environ.get("PORT", "8000"))
DEFAULT_DB_HOST = os.environ.get("DB_HOST", None)
DEFAULT_DB_PORT = os.environ.get("DB_PORT", None)
DEFAULT_DATABASE_TYPE = os.environ.get("DATABASE_TYPE", "chromadb").lower()
DEFAULT_DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.expanduser("~"))
DEFAULT_DBNAME = os.environ.get("DBNAME", "memory")
DEFAULT_USERNAME = os.environ.get("USERNAME", None)
DEFAULT_PASSWORD = os.environ.get("PASSWORD", None)
DEFAULT_API_TOKEN = os.environ.get("API_TOKEN", None)
DEFAULT_COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "memory")
DEFAULT_DOCUMENT_DIRECTORY = os.environ.get(
    "DOCUMENT_DIRECTORY", os.path.normpath("/documents")
)
DEFAULT_PROVIDER = os.getenv("PROVIDER", "openai")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "text-embedding-nomic-embed-text-v1.5")
DEFAULT_OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "http://host.docker.internal:1234/v1"
)
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "llama")


def initialize_retriever(
    db_type: str = DEFAULT_DATABASE_TYPE,
    db_path: str = DEFAULT_DATABASE_PATH,
    host: Optional[str] = DEFAULT_DB_HOST,
    port: Optional[str] = DEFAULT_DB_PORT,
    db_name: Optional[str] = DEFAULT_DBNAME,
    username: Optional[str] = DEFAULT_USERNAME,
    password: Optional[str] = DEFAULT_PASSWORD,
    api_token: Optional[str] = DEFAULT_API_TOKEN,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    ensure_collection_exists: bool = True,
) -> RAGRetriever:
    try:
        db_type_lower = db_type.strip().lower()
        if db_type_lower == "chromadb":
            from vector_mcp.retriever.chromadb_retriever import ChromaDBRetriever

            if host and port:
                retriever: RAGRetriever = ChromaDBRetriever(
                    host=host, port=int(port), collection_name=collection_name
                )
            else:
                retriever: RAGRetriever = ChromaDBRetriever(
                    path=os.path.join(db_path, db_name), collection_name=collection_name
                )
        elif db_type_lower == "pgvector":
            from vector_mcp.retriever.pgvector_retriever import PGVectorRetriever

            retriever: RAGRetriever = PGVectorRetriever(
                host=host,
                port=port,
                dbname=db_name,
                username=username,
                password=password,
                collection_name=collection_name,
            )
        elif db_type_lower == "qdrant":
            from vector_mcp.retriever.qdrant_retriever import QdrantRetriever

            # Construct location string
            location = ":memory:"
            if host:
                if host == ":memory:":
                    location = ":memory:"
                elif host.startswith("http"):
                    # If host is already a URL, use it
                    location = f"{host}:{port}" if port else host
                else:
                    # Assume http if not specified
                    location = (
                        f"http://{host}:{port}" if port else f"http://{host}:6333"
                    )

            # Note: QdrantRetriever currently accepts location.
            # API token handling would need updates to QdrantRetriever if required.
            retriever: RAGRetriever = QdrantRetriever(
                location=location, collection_name=collection_name
            )
        elif db_type_lower == "couchbase":
            from vector_mcp.retriever.couchbase_retriever import CouchbaseRetriever

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
            from vector_mcp.retriever.mongodb_retriever import MongoDBRetriever

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
        if not retriever.connect_database(
            collection_name=collection_name, ensure_exists=ensure_collection_exists
        ):
            raise RuntimeError(
                f"Failed to connect to {db_type} database or initialize index."
            )
        return retriever
    except Exception as e:
        logger.error(f"Failed to initialize Vector Database: {str(e)}")
        raise e


def register_tools(mcp: FastMCP):
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})

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
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: Optional[str] = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: Optional[str] = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: Optional[str] = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: Optional[str] = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: Optional[str] = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the collection to create or retrieve",
            default=DEFAULT_COLLECTION_NAME,
        ),
        overwrite: Optional[bool] = Field(
            description="Whether to overwrite the collection if it exists",
            default=False,
        ),
        document_directory: Optional[Union[Path, str]] = Field(
            description="Document directory to read documents from",
            default=DEFAULT_DOCUMENT_DIRECTORY,
        ),
        document_paths: Optional[Union[Path, str]] = Field(
            description="Document paths on the file system or URLs to read from",
            default=None,
        ),
        document_contents: Optional[List[str]] = Field(
            description="List of string contents to ingest directly", default=None
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
                "Document Contents": "Yes" if document_contents else "No",
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
                document_contents=document_contents,
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
            "title": "Vector Search Texts from a Collection",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"search"},
    )
    async def semantic_search(
        db_type: str = Field(
            description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: Optional[str] = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: Optional[str] = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: Optional[str] = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: Optional[str] = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: Optional[str] = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the collection to search",
            default=DEFAULT_COLLECTION_NAME,
        ),
        question: str = Field(
            description="The question or phrase to similarity search in the vector database",
            default=None,
        ),
        number_results: int = Field(
            description="The total number of searched document texts to provide",
            default=1,
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
            results = retriever.query(question=question, number_results=number_results)
            texts = "\n\n".join([r["text"] for r in results])
            if ctx:
                await ctx.report_progress(progress=100, total=100)
            response = {
                "searched_texts": texts,
                "message": "Collection searched from successfully",
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
            "title": "BM25 Search / Keyword Search",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"search"},
    )
    async def lexical_search(
        db_type: str = Field(
            description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: Optional[str] = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: Optional[str] = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: Optional[str] = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: Optional[str] = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: Optional[str] = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the collection to search",
            default=DEFAULT_COLLECTION_NAME,
        ),
        question: str = Field(
            description="The question or keyword to search in the vector database using BM25 or keyword matching",
            default=None,
        ),
        number_results: int = Field(
            description="The total number of searched document texts to provide",
            default=1,
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """This is a lexical or term based search that retrieves and gathers related knowledge from the database instance using the question variable via BM25.
        This provides a complementary search method to vector search, useful for exact keyword matching.
        """
        logger.debug(f"Initializing collection for BM25: {collection_name}")

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
            logger.debug(f"BM25 Querying collection: {question}")
            results = retriever.bm25_query(
                question=question, number_results=number_results
            )
            texts = "\n\n".join([r["text"] for r in results])
            if ctx:
                await ctx.report_progress(progress=100, total=100)
            response = {
                "searched_texts": texts,
                "message": "Collection searched successfully via BM25",
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
            logger.error(f"Invalid input for lexical_search: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to lexical_search: {str(e)}")
            raise RuntimeError(f"Failed to lexical_search: {str(e)}")

    @mcp.tool(
        annotations={
            "title": "Hybrid Search (Semantic + BM25)",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"search"},
    )
    async def search(
        db_type: str = Field(
            description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: Optional[str] = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: Optional[str] = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: Optional[str] = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: Optional[str] = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: Optional[str] = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the collection to search",
            default=DEFAULT_COLLECTION_NAME,
        ),
        question: str = Field(
            description="The question or phrase to hybrid search in the vector database",
            default=None,
        ),
        number_results: int = Field(
            description="The total number of hybrid searched document texts to provide",
            default=1,
        ),
        semantic_weight: float = Field(
            description="Weight for semantic results in fusion (0-1)",
            default=0.5,
        ),
        bm25_weight: float = Field(
            description="Weight for BM25 results in fusion (0-1)",
            default=0.5,
        ),
        rrf_k: int = Field(
            description="RRF constant (higher reduces bias toward top ranks)",
            default=60,
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """Performs a hybrid search combining semantic (vector) and lexical (BM25) methods.
        Retrieves results from both, merges them using weighted Reciprocal Rank Fusion (RRF),
        and returns the top combined results.
        """
        logger.debug(f"Initializing collection for hybrid: {collection_name}")

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

            # Fetch semantic results (assume returns list of dicts with 'text', 'score', 'id')
            semantic_results: List[Dict] = retriever.query(
                question=question,
                number_results=number_results * 2,  # Fetch extra for merging
            )

            # Fetch BM25 results
            bm25_results: List[Dict] = retriever.bm25_query(
                question=question, number_results=number_results * 2
            )

            if ctx:
                await ctx.report_progress(progress=50, total=100)

            # Merge and rerank with weighted RRF
            combined = {}
            for rank, res in enumerate(semantic_results, 1):
                # Fallback to hash if id not present
                doc_id = res.get("id") or hashlib.md5(res["text"].encode()).hexdigest()
                if doc_id not in combined:
                    combined[doc_id] = {"text": res["text"], "rrf_score": 0}
                combined[doc_id]["rrf_score"] += semantic_weight / (rank + rrf_k)

            for rank, res in enumerate(bm25_results, 1):
                doc_id = res.get("id") or hashlib.md5(res["text"].encode()).hexdigest()
                if doc_id not in combined:
                    combined[doc_id] = {"text": res["text"], "rrf_score": 0}
                combined[doc_id]["rrf_score"] += bm25_weight / (rank + rrf_k)

            # Sort by RRF score descending and take top N
            sorted_results = sorted(
                combined.values(), key=lambda x: x["rrf_score"], reverse=True
            )[:number_results]
            texts = [
                res["text"] for res in sorted_results
            ]  # Extract texts for consistency

            if ctx:
                await ctx.report_progress(progress=100, total=100)

            response = {
                "searched_texts": texts,
                "message": "Collection searched successfully via hybrid method",
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
            logger.error(f"Invalid input for search: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to search: {str(e)}")
            raise RuntimeError(f"Failed to search: {str(e)}")

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
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: Optional[str] = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: Optional[str] = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: Optional[str] = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: Optional[str] = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: Optional[str] = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the target collection.", default=None
        ),
        document_directory: Optional[Union[Path, str]] = Field(
            description="Document directory to read documents from",
            default=DEFAULT_DOCUMENT_DIRECTORY,
        ),
        document_paths: Optional[Union[Path, str]] = Field(
            description="Document paths on the file system or URLs to read from",
            default=None,
        ),
        document_contents: Optional[List[str]] = Field(
            description="List of string contents to ingest directly", default=None
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """Adds documents to an existing collection in the vector database.
        This can be used to extend collections with additional documents"""
        if not document_directory and not document_paths and not document_contents:
            raise ValueError(
                "At least one of document_directory, document_paths, or document_contents must be provided."
            )

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
            f"Inserting documents into collection: {collection_name}. "
            f"Directory: {document_directory}, Paths: {document_paths}, Contents: {'Yes' if document_contents else 'No'}"
        )

        try:
            if ctx:
                await ctx.report_progress(progress=0, total=100)
            texts = retriever.add_documents(
                document_directory=document_directory,
                document_paths=document_paths,
                document_contents=document_contents,
            )
            if ctx:
                await ctx.report_progress(progress=100, total=100)

            response = {
                "added_texts": texts,
                "message": "Collection created successfully",
                "data": {
                    "Database Type": db_type,
                    "Collection Name": collection_name,
                    "Document Directory": document_directory,
                    "Document Paths": document_paths,
                    "Document Contents": "Yes" if document_contents else "No",
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
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: Optional[str] = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: Optional[str] = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: Optional[str] = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: Optional[str] = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: Optional[str] = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the target collection.", default=None
        ),
        confirm: bool = Field(
            description="Explicitly confirm deletion without interactive prompt",
            default=False,
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """Deletes a collection from the vector database."""

        if not confirm:
            if ctx:
                message = f"Are you sure you want to DELETE collection {collection_name} from {db_type}?"
                try:
                    result = await ctx.elicit(message, response_type=bool)
                    if result.action != "accept" or not result.data:
                        return {
                            "status": "cancelled",
                            "message": "Operation cancelled by user.",
                        }
                except Exception as e:
                    logger.warning(f"Elicitation failed: {str(e)}")
                    return {
                        "status": "error",
                        "message": "Elicitation not supported by client. Please set 'confirm=True' to force deletion.",
                    }
            else:
                return {
                    "status": "error",
                    "message": "Context missing and confirm=False. Please set 'confirm=True' to force deletion.",
                }

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

    @mcp.tool(
        annotations={
            "title": "List Collections",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"collection_management"},
    )
    async def list_collections(
        db_type: str = Field(
            description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)",
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: Optional[str] = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: Optional[str] = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: Optional[str] = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: Optional[str] = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: Optional[str] = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> Dict:
        """Lists all collections in the vector database."""

        if db_type == "chromadb" and os.environ.get("DATABASE_TYPE"):
            db_type = os.environ.get("DATABASE_TYPE")

        try:
            retriever = initialize_retriever(
                db_type=db_type,
                db_path=db_path,
                host=host,
                port=port,
                db_name=db_name,
                username=username,
                password=password,
                ensure_collection_exists=False,
            )
            logger.debug(f"Listing collections for: {db_type}")

            if ctx:
                await ctx.report_progress(progress=0, total=100)

            collections = retriever.vector_db.get_collections()
            collection_names = []
            if isinstance(collections, list) or isinstance(collections, tuple):
                for c in collections:
                    if hasattr(c, "name"):
                        collection_names.append(c.name)
                    else:
                        collection_names.append(str(c))
            else:
                collection_names = str(collections)

            if ctx:
                await ctx.report_progress(progress=100, total=100)
            response = {
                "collections": collection_names,
                "message": "Collections listed successfully",
                "data": {
                    "Database Type": db_type,
                    "Database": db_name,
                    "Database Host": host,
                },
                "status": 200,
            }
            return response
        except ValueError as e:
            logger.error(f"Invalid input for list_collections: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to list collections: {str(e)}")


def vector_mcp():
    print(f"vector_mcp v{__version__}")
    parser = argparse.ArgumentParser(description="Vector MCP")
    parser.add_argument(
        "-t",
        "--transport",
        default=DEFAULT_TRANSPORT,
        choices=["stdio", "streamable-http", "sse"],
        help="Transport method: 'stdio', 'streamable-http', or 'sse' [legacy] (default: stdio)",
    )
    parser.add_argument(
        "-s",
        "--host",
        default=DEFAULT_HOST,
        help="Host address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
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
    parser.add_argument(
        "--token-algorithm",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM"),
        choices=[
            "HS256",
            "HS384",
            "HS512",
            "RS256",
            "RS384",
            "RS512",
            "ES256",
            "ES384",
            "ES512",
        ],
        help="JWT signing algorithm (required for HMAC or static key). Auto-detected for JWKS.",
    )
    parser.add_argument(
        "--token-secret",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Shared secret for HMAC (HS*) or PEM public key for static asymmetric verification.",
    )
    parser.add_argument(
        "--token-public-key",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Path to PEM public key file or inline PEM string (for static asymmetric keys).",
    )
    parser.add_argument(
        "--required-scopes",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES"),
        help="Comma-separated list of required scopes (e.g., gitlab.read,gitlab.write).",
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
    # Delegation params
    parser.add_argument(
        "--enable-delegation",
        action="store_true",
        default=to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
        help="Enable OIDC token delegation",
    )
    parser.add_argument(
        "--audience",
        default=os.environ.get("AUDIENCE", None),
        help="Audience for the delegated token",
    )
    parser.add_argument(
        "--delegated-scopes",
        default=os.environ.get("DELEGATED_SCOPES", "api"),
        help="Scopes for the delegated token (space-separated)",
    )
    parser.add_argument(
        "--openapi-file",
        default=None,
        help="Path to the OpenAPI JSON file to import additional tools from",
    )
    parser.add_argument(
        "--openapi-base-url",
        default=None,
        help="Base URL for the OpenAPI client (overrides instance URL)",
    )
    parser.add_argument(
        "--openapi-use-token",
        action="store_true",
        help="Use the incoming Bearer token (from MCP request) to authenticate OpenAPI import",
    )

    parser.add_argument(
        "--openapi-username",
        default=os.getenv("OPENAPI_USERNAME"),
        help="Username for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-password",
        default=os.getenv("OPENAPI_PASSWORD"),
        help="Password for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-id",
        default=os.getenv("OPENAPI_CLIENT_ID"),
        help="OAuth client ID for OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-secret",
        default=os.getenv("OPENAPI_CLIENT_SECRET"),
        help="OAuth client secret for OpenAPI import",
    )

    args = parser.parse_args()

    if args.port < 0 or args.port > 65535:
        print(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)

    # Update config with CLI arguments
    config["enable_delegation"] = args.enable_delegation
    config["audience"] = args.audience or config["audience"]
    config["delegated_scopes"] = args.delegated_scopes or config["delegated_scopes"]
    config["oidc_config_url"] = args.oidc_config_url or config["oidc_config_url"]
    config["oidc_client_id"] = args.oidc_client_id or config["oidc_client_id"]
    config["oidc_client_secret"] = (
        args.oidc_client_secret or config["oidc_client_secret"]
    )

    # Configure delegation if enabled
    if config["enable_delegation"]:
        if args.auth_type != "oidc-proxy":
            logger.error("Token delegation requires auth-type=oidc-proxy")
            sys.exit(1)
        if not config["audience"]:
            logger.error("audience is required for delegation")
            sys.exit(1)
        if not all(
            [
                config["oidc_config_url"],
                config["oidc_client_id"],
                config["oidc_client_secret"],
            ]
        ):
            logger.error(
                "Delegation requires complete OIDC configuration (oidc-config-url, oidc-client-id, oidc-client-secret)"
            )
            sys.exit(1)

        # Fetch OIDC configuration to get token_endpoint
        try:
            logger.info(
                "Fetching OIDC configuration",
                extra={"oidc_config_url": config["oidc_config_url"]},
            )
            oidc_config_resp = requests.get(config["oidc_config_url"])
            oidc_config_resp.raise_for_status()
            oidc_config = oidc_config_resp.json()
            config["token_endpoint"] = oidc_config.get("token_endpoint")
            if not config["token_endpoint"]:
                logger.error("No token_endpoint found in OIDC configuration")
                raise ValueError("No token_endpoint found in OIDC configuration")
            logger.info(
                "OIDC configuration fetched successfully",
                extra={"token_endpoint": config["token_endpoint"]},
            )
        except Exception as e:
            print(f"Failed to fetch OIDC configuration: {e}")
            logger.error(
                "Failed to fetch OIDC configuration",
                extra={"error_type": type(e).__name__, "error_message": str(e)},
            )
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
        auth = StaticTokenVerifier(
            tokens={
                "test-token": {"client_id": "test-user", "scopes": ["read", "write"]},
                "admin-token": {"client_id": "admin", "scopes": ["admin"]},
            }
        )
    elif args.auth_type == "jwt":
        # Fallback to env vars if not provided via CLI
        jwks_uri = args.token_jwks_uri or os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI")
        issuer = args.token_issuer or os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER")
        audience = args.token_audience or os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE")
        algorithm = args.token_algorithm
        secret_or_key = args.token_secret or args.token_public_key
        public_key_pem = None

        if not (jwks_uri or secret_or_key):
            logger.error(
                "JWT auth requires either --token-jwks-uri or --token-secret/--token-public-key"
            )
            sys.exit(1)
        if not (issuer and audience):
            logger.error("JWT requires --token-issuer and --token-audience")
            sys.exit(1)

        # Load static public key from file if path is given
        if args.token_public_key and os.path.isfile(args.token_public_key):
            try:
                with open(args.token_public_key, "r") as f:
                    public_key_pem = f.read()
                logger.info(f"Loaded static public key from {args.token_public_key}")
            except Exception as e:
                print(f"Failed to read public key file: {e}")
                logger.error(f"Failed to read public key file: {e}")
                sys.exit(1)
        elif args.token_public_key:
            public_key_pem = args.token_public_key  # Inline PEM

        # Validation: Conflicting options
        if jwks_uri and (algorithm or secret_or_key):
            logger.warning(
                "JWKS mode ignores --token-algorithm and --token-secret/--token-public-key"
            )

        # HMAC mode
        if algorithm and algorithm.startswith("HS"):
            if not secret_or_key:
                logger.error(f"HMAC algorithm {algorithm} requires --token-secret")
                sys.exit(1)
            if jwks_uri:
                logger.error("Cannot use --token-jwks-uri with HMAC")
                sys.exit(1)
            public_key = secret_or_key
        else:
            public_key = public_key_pem

        # Required scopes
        required_scopes = None
        if args.required_scopes:
            required_scopes = [
                s.strip() for s in args.required_scopes.split(",") if s.strip()
            ]

        try:
            auth = JWTVerifier(
                jwks_uri=jwks_uri,
                public_key=public_key,
                issuer=issuer,
                audience=audience,
                algorithm=(
                    algorithm if algorithm and algorithm.startswith("HS") else None
                ),
                required_scopes=required_scopes,
            )
            logger.info(
                "JWTVerifier configured",
                extra={
                    "mode": (
                        "JWKS"
                        if jwks_uri
                        else (
                            "HMAC"
                            if algorithm and algorithm.startswith("HS")
                            else "Static Key"
                        )
                    ),
                    "algorithm": algorithm,
                    "required_scopes": required_scopes,
                },
            )
        except Exception as e:
            print(f"Failed to initialize JWTVerifier: {e}")
            logger.error(f"Failed to initialize JWTVerifier: {e}")
            sys.exit(1)
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
                "oauth-proxy requires oauth-upstream-auth-endpoint, oauth-upstream-token-endpoint, "
                "oauth-upstream-client-id, oauth-upstream-client-secret, oauth-base-url, token-jwks-uri, "
                "token-issuer, token-audience"
            )
            logger.error(
                "oauth-proxy requires oauth-upstream-auth-endpoint, oauth-upstream-token-endpoint, "
                "oauth-upstream-client-id, oauth-upstream-client-secret, oauth-base-url, token-jwks-uri, "
                "token-issuer, token-audience",
                extra={
                    "auth_endpoint": args.oauth_upstream_auth_endpoint,
                    "token_endpoint": args.oauth_upstream_token_endpoint,
                    "client_id": args.oauth_upstream_client_id,
                    "base_url": args.oauth_base_url,
                    "jwks_uri": args.token_jwks_uri,
                    "issuer": args.token_issuer,
                    "audience": args.token_audience,
                },
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
            logger.error(
                "oidc-proxy requires oidc-config-url, oidc-client-id, oidc-client-secret, oidc-base-url",
                extra={
                    "config_url": args.oidc_config_url,
                    "client_id": args.oidc_client_id,
                    "base_url": args.oidc_base_url,
                },
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
            logger.error(
                "remote-oauth requires remote-auth-servers, remote-base-url, token-jwks-uri, token-issuer, token-audience",
                extra={
                    "auth_servers": args.remote_auth_servers,
                    "base_url": args.remote_base_url,
                    "jwks_uri": args.token_jwks_uri,
                    "issuer": args.token_issuer,
                    "audience": args.token_audience,
                },
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

    # === 2. Build Middleware List ===
    middlewares: List[
        Union[
            UserTokenMiddleware,
            ErrorHandlingMiddleware,
            RateLimitingMiddleware,
            TimingMiddleware,
            LoggingMiddleware,
            JWTClaimsLoggingMiddleware,
            EunomiaMcpMiddleware,
        ]
    ] = [
        ErrorHandlingMiddleware(include_traceback=True, transform_errors=True),
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20),
        TimingMiddleware(),
        LoggingMiddleware(),
        JWTClaimsLoggingMiddleware(),
    ]
    if config["enable_delegation"] or args.auth_type == "jwt":
        middlewares.insert(0, UserTokenMiddleware(config=config))  # Must be first

    if args.eunomia_type in ["embedded", "remote"]:
        try:
            from eunomia_mcp import create_eunomia_middleware

            policy_file = args.eunomia_policy_file or "mcp_policies.json"
            eunomia_endpoint = (
                args.eunomia_remote_url if args.eunomia_type == "remote" else None
            )
            eunomia_mw = create_eunomia_middleware(
                policy_file=policy_file, eunomia_endpoint=eunomia_endpoint
            )
            middlewares.append(eunomia_mw)
            logger.info(f"Eunomia middleware enabled ({args.eunomia_type})")
        except Exception as e:
            print(f"Failed to load Eunomia middleware: {e}")
            logger.error("Failed to load Eunomia middleware", extra={"error": str(e)})
            sys.exit(1)

    mcp = FastMCP(name="VectorMCP", auth=auth)
    register_tools(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    print("\nStarting Vector MCP Server")
    print(f"  Transport: {args.transport.upper()}")
    print(f"  Auth: {args.auth_type}")
    print(f"  Delegation: {'ON' if config['enable_delegation'] else 'OFF'}")
    print(f"  Eunomia: {args.eunomia_type}")

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


if __name__ == "__main__":
    vector_mcp()
