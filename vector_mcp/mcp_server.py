#!/usr/bin/python
import warnings

# Filter RequestsDependencyWarning early to prevent log spam
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

# General urllib3/chardet mismatch warnings
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

from agent_utilities.base_utilities import to_boolean, to_integer
from agent_utilities.mcp_utilities import (
    create_mcp_server,
    ctx_log,
)
from dotenv import find_dotenv, load_dotenv
from fastmcp import Context, FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from vector_mcp.retriever.retriever import RAGRetriever

__version__ = "1.1.59"

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger("VectorServer")

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
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "text-embedding-nomic-embed-text-v2-moe")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "llama")


from llama_index.core import Settings

chunk_size = to_integer(os.getenv("CHUNK_SIZE", "1024"))
Settings.chunk_size = chunk_size
logger.info(f"Global chunk size set to: {chunk_size}")


DEFAULT_COLLECTIONS = [
    "decisions",
    "user",
    "myself",
    "knowledge",
    "tasks",
    "patterns",
]


def create_default_collections(
    db_type: str = DEFAULT_DATABASE_TYPE,
    db_path: str = DEFAULT_DATABASE_PATH,
    host: str | None = DEFAULT_DB_HOST,
    port: str | None = DEFAULT_DB_PORT,
    db_name: str | None = DEFAULT_DBNAME,
    username: str | None = DEFAULT_USERNAME,
    password: str | None = DEFAULT_PASSWORD,
):
    for collection in DEFAULT_COLLECTIONS:
        try:
            initialize_retriever(
                db_type=db_type,
                db_path=db_path,
                host=host,
                port=port,
                db_name=db_name,
                username=username,
                password=password,
                collection_name=collection,
                ensure_collection_exists=True,
            )
            logger.info(f"Ensured default collection exists: {collection}")
        except Exception as e:
            logger.error(f"Failed to create default collection {collection}: {e}")


def initialize_retriever(
    db_type: str = DEFAULT_DATABASE_TYPE,
    db_path: str = DEFAULT_DATABASE_PATH,
    host: str | None = DEFAULT_DB_HOST,
    port: str | None = DEFAULT_DB_PORT,
    db_name: str | None = DEFAULT_DBNAME,
    username: str | None = DEFAULT_USERNAME,
    password: str | None = DEFAULT_PASSWORD,
    _api_token: str | None = DEFAULT_API_TOKEN,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    ensure_collection_exists: bool = True,
) -> RAGRetriever:
    try:
        db_type_lower = db_type.strip().lower()
        retriever: RAGRetriever
        if db_type_lower == "chromadb":
            from vector_mcp.retriever.chromadb_retriever import ChromaDBRetriever

            if host and port:
                retriever = ChromaDBRetriever(
                    host=host, port=int(port), collection_name=collection_name
                )
            else:
                retriever = ChromaDBRetriever(
                    path=os.path.join(db_path, db_name or ""),
                    collection_name=collection_name,
                )
        elif db_type_lower == "postgres":
            from vector_mcp.retriever.postgres_retriever import PGVectorRetriever

            retriever = PGVectorRetriever(
                host=host,
                port=port,
                dbname=db_name or "",
                username=username or "",
                password=password or "",
                collection_name=collection_name,
            )
        elif db_type_lower == "qdrant":
            from vector_mcp.retriever.qdrant_retriever import QdrantRetriever

            location = ":memory:"
            if host:
                if host == ":memory:":
                    location = ":memory:"
                elif host.startswith("http"):
                    location = f"{host}:{port}" if port else host
                else:
                    location = (
                        f"http://{host}:{port}" if port else f"http://{host}:6333"
                    )

            retriever = QdrantRetriever(
                location=location, collection_name=collection_name
            )
        elif db_type_lower == "couchbase":
            from vector_mcp.retriever.couchbase_retriever import CouchbaseRetriever

            connection_string = (
                f"couchbase://{host}" if host else "couchbase://localhost"
            )
            if port:
                connection_string += f":{port}"
            retriever = CouchbaseRetriever(
                connection_string=connection_string,
                username=username or "Administrator",
                password=password or "password",
                bucket_name=db_name or "vector_db",
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
            retriever = MongoDBRetriever(
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


def register_misc_tools(mcp: FastMCP):
    pass
    pass

    async def health_check(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})


def register_collection_management_tools(mcp: FastMCP):
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
            description="Type of vector database (chromadb, postgres, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: str | None = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: str | None = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: str | None = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: str | None = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: str | None = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the collection to create or retrieve",
            default=DEFAULT_COLLECTION_NAME,
        ),
        overwrite: bool | None = Field(
            description="Whether to overwrite the collection if it exists",
            default=False,
        ),
        document_directory: Path | str | None = Field(
            description="Document directory to read documents from",
            default=DEFAULT_DOCUMENT_DIRECTORY,
        ),
        document_paths: Path | str | None = Field(
            description="Document paths on the file system or URLs to read from",
            default=None,
        ),
        document_contents: list[str] | None = Field(
            description="List of string contents to ingest directly", default=None
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> dict:
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

        ctx_log(ctx, logger, "debug",
            f"Creating collection: {collection_name}, overwrite: {overwrite}, "
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
                document_paths=[document_paths]
                if isinstance(document_paths, (str, Path))
                else document_paths,
                document_contents=document_contents,
            )
            if ctx:
                await ctx.report_progress(progress=100, total=100)
            response["completion"] = coll
            return response
        except ValueError as e:
            ctx_log(ctx, logger, "error", f"Invalid input for create_collection: {str(e)}")
            raise
        except Exception as e:
            ctx_log(ctx, logger, "error", f"Failed to create collection: {str(e)}")
            raise RuntimeError(f"Failed to create collection: {str(e)}") from e

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
            description="Type of vector database (chromadb, postgres, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: str | None = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: str | None = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: str | None = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: str | None = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: str | None = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the target collection.",
            default=DEFAULT_COLLECTION_NAME,
        ),
        document_directory: Path | str | None = Field(
            description="Document directory to read documents from",
            default=DEFAULT_DOCUMENT_DIRECTORY,
        ),
        document_paths: Path | str | None = Field(
            description="Document paths on the file system or URLs to read from",
            default=None,
        ),
        document_contents: list[str] | None = Field(
            description="List of string contents to ingest directly", default=None
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> dict:
        """Adds documents to an existing collection in the vector database.
        This can be used to extend collections with additional documents"""

        if not document_directory and not document_paths and not document_contents:
            raise ValueError(
                "At least one of document_directory, document_paths, or document_contents must be provided."
            )

        if document_directory:
            doc_dir_path = Path(document_directory)
            if doc_dir_path.exists() and doc_dir_path.is_dir():
                files = [f for f in doc_dir_path.iterdir() if f.is_file()]
                if not files and not document_paths and not document_contents:
                    ctx_log(ctx, logger, "warning", f"No files found in {document_directory}")
                    return {
                        "added_texts": [],
                        "message": "No documents found to ingest.",
                        "data": {
                            "Database Type": db_type,
                            "Collection Name": collection_name,
                            "Document Directory": document_directory,
                            "Status": "Skipped - Empty Directory",
                        },
                        "status": 200,
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
        ctx_log(ctx, logger, "debug",
            f"Inserting documents into collection: {collection_name}. "
            f"Directory: {document_directory}, Paths: {document_paths}, Contents: {'Yes' if document_contents else 'No'}"
        )

        try:
            if ctx:
                await ctx.report_progress(progress=0, total=100)
            texts = retriever.add_documents(
                document_directory=document_directory,
                document_paths=[document_paths]
                if isinstance(document_paths, (str, Path))
                else document_paths,
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
            ctx_log(ctx, logger, "error", f"Invalid input for insert_documents: {str(e)}")
            raise
        except Exception as e:
            ctx_log(ctx, logger, "error", f"Failed to insert documents: {str(e)}")
            raise RuntimeError(f"Failed to insert documents: {str(e)}") from e

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
            description="Type of vector database (chromadb, postgres, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: str | None = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: str | None = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: str | None = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: str | None = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: str | None = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the target collection.",
            default=DEFAULT_COLLECTION_NAME,
        ),
        confirm: bool = Field(
            description="Explicitly confirm deletion without interactive prompt",
            default=False,
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> dict:
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
                    ctx_log(ctx, logger, "warning", f"Elicitation failed: {str(e)}")
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
        ctx_log(ctx, logger, "debug", f"Deleting collection: {collection_name} from: {db_type}")

        try:
            if ctx:
                await ctx.report_progress(progress=0, total=100)
            assert retriever.vector_db is not None
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
            ctx_log(ctx, logger, "error", f"Invalid input for delete collection: {str(e)}")
            raise
        except Exception as e:
            ctx_log(ctx, logger, "error", f"Failed to delete collection: {str(e)}")
            raise RuntimeError(f"Failed to delete collection: {str(e)}") from e

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
            description="Type of vector database (chromadb, postgres, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: str | None = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: str | None = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: str | None = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: str | None = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: str | None = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> dict:
        """Lists all collections in the vector database."""

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
            ctx_log(ctx, logger, "debug", f"Listing collections for: {db_type}")

            if ctx:
                await ctx.report_progress(progress=0, total=100)

            assert retriever.vector_db is not None
            collections = retriever.vector_db.get_collections()
            collection_names = []
            if isinstance(collections, list) or isinstance(collections, tuple):
                for c in collections:
                    if hasattr(c, "name"):
                        collection_names.append(c.name)
                    else:
                        collection_names.append(str(c))
            else:
                collection_names = [str(collections)]

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
            ctx_log(ctx, logger, "error", f"Invalid input for list_collections: {str(e)}")
            raise
        except Exception as e:
            ctx_log(ctx, logger, "error", f"Failed to list collections: {str(e)}")
            import traceback

            ctx_log(ctx, logger, "error", traceback.format_exc())
            raise RuntimeError(traceback.format_exc()) from e


def register_search_tools(mcp: FastMCP):
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
            description="Type of vector database (chromadb, postgres, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: str | None = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: str | None = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: str | None = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: str | None = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: str | None = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the collection to search",
            default=DEFAULT_COLLECTION_NAME,
        ),
        question: str = Field(
            description="The question or phrase to similarity search in the vector database",
            default="",
        ),
        number_results: int = Field(
            description="The total number of searched document texts to provide",
            default=1,
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> dict:
        """Retrieves and gathers related knowledge from the vector database instance using the question variable.
        This can be used as a primary source of knowledge retrieval.
        It will return relevant text(s) which should be parsed for the most
        relevant information pertaining to the question and summarized as the final output
        """
        ctx_log(ctx, logger, "debug", f"Initializing collection: {collection_name}")

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
            ctx_log(ctx, logger, "debug", f"Querying collection: {question}")
            results = retriever.query(question=question, number_results=number_results)
            texts = "\n".join([r["text"] for r in results])
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
            ctx_log(ctx, logger, "error", f"Invalid input for get_collection: {str(e)}")
            raise
        except Exception as e:
            ctx_log(ctx, logger, "error", f"Failed to get collection: {str(e)}")
            raise RuntimeError(f"Failed to get collection: {str(e)}") from e

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
            description="Type of vector database (chromadb, postgres, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: str | None = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: str | None = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: str | None = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: str | None = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: str | None = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the collection to search",
            default=DEFAULT_COLLECTION_NAME,
        ),
        question: str = Field(
            description="The question or keyword to search in the vector database using BM25 or keyword matching",
            default="",
        ),
        number_results: int = Field(
            description="The total number of searched document texts to provide",
            default=1,
        ),
        ctx: Context = Field(
            description="FastMCP context for progress reporting", default=None
        ),
    ) -> dict:
        """This is a lexical or term based search that retrieves and gathers related knowledge from the database instance using the question variable via BM25.
        This provides a complementary search method to vector search, useful for exact keyword matching.
        """
        ctx_log(ctx, logger, "debug", f"Initializing collection for BM25: {collection_name}")

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
            ctx_log(ctx, logger, "debug", f"BM25 Querying collection: {question}")
            results = retriever.bm25_query(
                question=question, number_results=number_results
            )
            texts = "\n".join([r["text"] for r in results])
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
            ctx_log(ctx, logger, "error", f"Invalid input for lexical_search: {str(e)}")
            raise
        except Exception as e:
            ctx_log(ctx, logger, "error", f"Failed to lexical_search: {str(e)}")
            raise RuntimeError(f"Failed to lexical_search: {str(e)}") from e

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
            description="Type of vector database (chromadb, postgres, qdrant, couchbase, mongodb)",
            default=DEFAULT_DATABASE_TYPE,
        ),
        db_path: str = Field(
            description="The path to store chromadb files",
            default=DEFAULT_DATABASE_PATH,
        ),
        host: str | None = Field(
            description="Hostname or IP address of the database server",
            default=DEFAULT_DB_HOST,
        ),
        port: str | None = Field(
            description="Port number of the database server", default=DEFAULT_DB_PORT
        ),
        db_name: str | None = Field(
            description="Name of the database or path (depending on DB type)",
            default=DEFAULT_DBNAME,
        ),
        username: str | None = Field(
            description="Username for database authentication", default=DEFAULT_USERNAME
        ),
        password: str | None = Field(
            description="Password for database authentication", default=DEFAULT_PASSWORD
        ),
        collection_name: str = Field(
            description="Name of the collection to search",
            default=DEFAULT_COLLECTION_NAME,
        ),
        question: str = Field(
            description="The question or phrase to hybrid search in the vector database",
            default="",
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
    ) -> dict:
        """Performs a hybrid search combining semantic (vector) and lexical (BM25) methods.
        Retrieves results from both, merges them using weighted Reciprocal Rank Fusion (RRF),
        and returns the top combined results.
        """
        ctx_log(ctx, logger, "debug", f"Initializing collection for hybrid: {collection_name}")

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

            semantic_results: list[dict] = retriever.query(
                question=question,
                number_results=number_results * 2,
            )

            bm25_results: list[dict] = retriever.bm25_query(
                question=question, number_results=number_results * 2
            )

            if ctx:
                await ctx.report_progress(progress=50, total=100)

            combined = {}
            for rank, res in enumerate(semantic_results, 1):
                doc_id = res.get("id") or hashlib.md5(res["text"].encode()).hexdigest()
                if doc_id not in combined:
                    combined[doc_id] = {"text": res["text"], "rrf_score": 0}
                combined[doc_id]["rrf_score"] += semantic_weight / (rank + rrf_k)

            for rank, res in enumerate(bm25_results, 1):
                doc_id = res.get("id") or hashlib.md5(res["text"].encode()).hexdigest()
                if doc_id not in combined:
                    combined[doc_id] = {"text": res["text"], "rrf_score": 0}
                combined[doc_id]["rrf_score"] += bm25_weight / (rank + rrf_k)

            sorted_results = sorted(
                combined.values(), key=lambda x: x["rrf_score"], reverse=True
            )[:number_results]
            texts = [res["text"] for res in sorted_results]

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
            ctx_log(ctx, logger, "error", f"Invalid input for search: {str(e)}")
            raise
        except Exception as e:
            ctx_log(ctx, logger, "error", f"Failed to search: {str(e)}")
            raise RuntimeError(f"Failed to search: {str(e)}") from e


def get_mcp_instance() -> tuple[Any, Any, Any, Any]:
    """Initialize and return the MCP instance, args, and middlewares."""
    load_dotenv(find_dotenv())

    args, mcp, middlewares = create_mcp_server(
        name="VectorMCP",
        version=__version__,
        instructions="Vector MCP Server — Semantic knowledge retrieval and collection management using vector databases.",
    )

    DEFAULT_MISCTOOL = to_boolean(os.getenv("MISCTOOL", "True"))
    if DEFAULT_MISCTOOL:
        register_misc_tools(mcp)
    DEFAULT_COLLECTION_MANAGEMENTTOOL = to_boolean(
        os.getenv("COLLECTION_MANAGEMENTTOOL", "True")
    )
    if DEFAULT_COLLECTION_MANAGEMENTTOOL:
        register_collection_management_tools(mcp)
    DEFAULT_SEARCHTOOL = to_boolean(os.getenv("SEARCHTOOL", "True"))
    if DEFAULT_SEARCHTOOL:
        register_search_tools(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)
    registered_tags: list[str] = []
    return mcp, args, middlewares, registered_tags


def mcp_server() -> None:
    mcp, args, middlewares, registered_tags = get_mcp_instance()
    print(f"{'vector-mcp'} MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)
    print(f"  Dynamic Tags Loaded: {len(set(registered_tags))}", file=sys.stderr)

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
    mcp_server()
