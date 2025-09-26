#!/usr/bin/python
# coding: utf-8
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Union
from fastmcp import FastMCP, Context
from pydantic import Field
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

    args = parser.parse_args()

    if args.port < 0 or args.port > 65535:
        print(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Transport not supported")
        sys.exit(1)


if __name__ == "__main__":
    vector_mcp()
