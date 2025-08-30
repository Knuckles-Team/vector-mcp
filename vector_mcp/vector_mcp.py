#!/usr/bin/python
# coding: utf-8
import getopt
import os
import sys
import logging
from typing import Optional, List, Dict, Any, Union
from fastmcp import FastMCP, Context
from pydantic import Field
from .base import Document, VectorDB  # Adjust import path as needed
from .pgvector import PGVectorDB  # Adjust import path as needed
from .qdrant import QdrantVectorDB  # Adjust import path as needed
from .couchbase import CouchbaseVectorDB  # Adjust import path as needed
from .mongodb import MongoDBAtlasVectorDB  # Adjust import path as needed
from .chromadb import ChromaVectorDB  # Adjust import path as needed
from .utils import get_logger  # Adjust import path as needed

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = get_logger("VectorServer")

environment_host = os.environ.get("HOST", "localhost")
environment_port = os.environ.get("PORT", 5432)
environment_db_name = os.environ.get("DBNAME", "postgres")
environment_username = os.environ.get("USERNAME", "postgres")
environment_password = os.environ.get("PASSWORD", "")
environment_api_token = os.environ.get("API_TOKEN", "")


mcp = FastMCP(name="PGVectorServer")
mcp.on_duplicate_tools = "error"


def initialize_database(
        db_type: str = Field(description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)", default="chromadb"),
        host: Optional[str] = Field(description="Hostname or IP address of the database server", default=environment_host),
        port: Optional[str] = Field(description="Port number of the database server", default=environment_port),
        db_name: Optional[str] = Field(description="Name of the database or path (depending on DB type)", default=environment_db_name),
        username: Optional[str] = Field(description="Username for database authentication", default=environment_username),
        password: Optional[str] = Field(description="Password for database authentication", default=environment_password),
        api_token: Optional[str] = Field(description="API Token for database authentication", default=environment_api_token)
) -> VectorDB:
    try:
        db_type_lower = db_type.strip().lower()
        if db_type_lower == "chromadb":
            if host and port:
                db: VectorDB = ChromaVectorDB(host=host, port=int(port))
            else:
                db: VectorDB = ChromaVectorDB(path=db_name or "tmp/db")
        elif db_type_lower == "pgvector":
            db: VectorDB = PGVectorDB(
                host=host,
                port=port,
                dbname=db_name,
                username=username,
                password=password,
            )
        elif db_type_lower == "qdrant":
            client_kwargs = {}
            if host:
                client_kwargs = {"host": host} if host else {"location": ":memory:"}
            if port:
                client_kwargs["port"] = str(port)
            if password:
                client_kwargs["api_key"] = api_token
            db: VectorDB = QdrantVectorDB(client_kwargs=client_kwargs)
        elif db_type_lower == "couchbase":
            connection_string = f"couchbase://{host}" if host else "couchbase://localhost"
            if port:
                connection_string += f":{port}"
            db: VectorDB = CouchbaseVectorDB(
                connection_string=connection_string,
                username=username,
                password=password,
                bucket_name=db_name or "vector_db",
            )
        elif db_type_lower == "mongodb":
            connection_string = ""
            if host:
                connection_string = f"mongodb://{username}:{password}@{host}:{port or '27017'}/{db_name}" if username and password else f"mongodb://{host}:{port or '27017'}/{db_name}"
            db: VectorDB = MongoDBAtlasVectorDB(
                connection_string=connection_string,
                database_name=db_name or "vector_db",
            )
        else:
            logger.error(f"Failed to identify vector database from supported databases")
            sys.exit(1)
        logger.info("Vector Database initialized successfully.")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize Vector Database: {str(e)}")
        sys.exit(1)


@mcp.tool(tags={"collection_management"})
async def create_collection(
        db_type: str = Field(description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)", default="chromadb"),
        host: Optional[str] = Field(description="Hostname or IP address of the database server", default=environment_host),
        port: Optional[str] = Field(description="Port number of the database server", default=environment_port),
        db_name: Optional[str] = Field(description="Name of the database or path (depending on DB type)", default=environment_db_name),
        username: Optional[str] = Field(description="Username for database authentication", default=environment_username),
        password: Optional[str] = Field(description="Password for database authentication", default=environment_password),
        collection_name: str = Field(description="Name of the collection to create or retrieve", default="memory"),
        overwrite: Optional[bool] = Field(description="Whether to overwrite the collection if it exists", default=False),
        get_or_create: Optional[bool] = Field(description="Whether to retrieve the collection if it exists instead of raising an error", default=True),
        ctx: Context = Field(description="FastMCP context for progress reporting", default=None),
) -> str:
    """Creates a new collection or retrieves an existing one in the vector database."""
    if not collection_name:
        raise ValueError("collection_name must not be empty")

    db = initialize_database(db_type=db_type, host=host, port=port, db_name=db_name, username=username, password=password)

    logger.debug(f"Creating collection: {collection_name}, overwrite: {overwrite}, get_or_create: {get_or_create}")

    try:
        if ctx:
            await ctx.report_progress(progress=0, total=100)
        coll = db.create_collection(collection_name, overwrite, get_or_create)
        if ctx:
            await ctx.report_progress(progress=100, total=100)
        return f"Collection '{collection_name}' created or retrieved successfully.\n{coll}"
    except ValueError as e:
        logger.error(f"Invalid input for create_collection: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        raise RuntimeError(f"Failed to create collection: {str(e)}")

@mcp.tool(tags={"collection_management"})
async def get_collection(
        db_type: str = Field(description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)", default="chromadb"),
        host: Optional[str] = Field(description="Hostname or IP address of the database server", default=environment_host),
        port: Optional[str] = Field(description="Port number of the database server", default=environment_port),
        db_name: Optional[str] = Field(description="Name of the database or path (depending on DB type)", default=environment_db_name),
        username: Optional[str] = Field(description="Username for database authentication", default=environment_username),
        password: Optional[str] = Field(description="Password for database authentication", default=environment_password),
        collection_name: str = Field(description="Name of the collection to retrieve", default="memory"),
        ctx: Context = Field(description="FastMCP context for progress reporting", default=None),
) -> str:
    """Retrieves a collection from the vector database."""
    logger.debug(f"Getting collection: {collection_name}")

    db = initialize_database(db_type=db_type, host=host, port=port, db_name=db_name, username=username, password=password)

    try:
        if ctx:
            await ctx.report_progress(progress=0, total=100)
        coll = db.get_collection(collection_name)
        if ctx:
            await ctx.report_progress(progress=100, total=100)
        coll_name = coll.name if hasattr(coll, 'name') else str(coll)
        return f"Collection '{coll_name}' retrieved successfully."
    except ValueError as e:
        logger.error(f"Invalid input for get_collection: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to get collection: {str(e)}")
        raise RuntimeError(f"Failed to get collection: {str(e)}")

@mcp.tool(tags={"document_management"})
async def insert_docs(
        db_type: str = Field(description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)", default="chromadb"),
        host: Optional[str] = Field(description="Hostname or IP address of the database server", default=environment_host),
        port: Optional[str] = Field(description="Port number of the database server", default=environment_port),
        db_name: Optional[str] = Field(description="Name of the database or path (depending on DB type)", default=environment_db_name),
        username: Optional[str] = Field(description="Username for database authentication", default=environment_username),
        password: Optional[str] = Field(description="Password for database authentication", default=environment_password),
        docs: List[Union[Document, Dict[str, Any]]] = Field(description="List of documents, each as a dict with 'id' (str|int), 'content' (str), optional 'metadata' (dict), 'embedding' (list[float|int]).", default=None),
        collection_name: str = Field(description="Name of the target collection.", default=None),
        upsert: Optional[bool] = Field(description="If True, update documents if they exist.", default=False),
        ctx: Context = Field(description="FastMCP context for progress reporting", default=None),
) -> str:
    """Inserts documents into a collection in the vector database."""
    if not docs:
        raise ValueError("docs list must not be empty")
    if not collection_name:
        raise ValueError("collection_name must not be empty")
    for doc in docs:
        if 'id' not in doc or 'content' not in doc:
            raise ValueError("Each document must have 'id' and 'content'")

    db = initialize_database(db_type=db_type, host=host, port=port, db_name=db_name, username=username, password=password)
    logger.debug(f"Inserting {len(docs)} documents into collection: {collection_name}, upsert: {upsert}")

    try:
        if ctx:
            await ctx.report_progress(progress=0, total=100)
        db.insert_docs(docs, collection_name, upsert)
        if ctx:
            await ctx.report_progress(progress=100, total=100)
        return f"{len(docs)} documents inserted into '{collection_name}' successfully."
    except ValueError as e:
        logger.error(f"Invalid input for insert_docs: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to insert documents: {str(e)}")
        raise RuntimeError(f"Failed to insert documents: {str(e)}")

@mcp.tool(tags={"document_management"})
async def update_docs(
        db_type: str = Field(description="Type of vector database (chromadb, pgvector, qdrant, couchbase, mongodb)", default="chromadb"),
        host: Optional[str] = Field(description="Hostname or IP address of the database server", default=environment_host),
        port: Optional[str] = Field(description="Port number of the database server", default=environment_port),
        db_name: Optional[str] = Field(description="Name of the database or path (depending on DB type)", default=environment_db_name),
        username: Optional[str] = Field(description="Username for database authentication", default=environment_username),
        password: Optional[str] = Field(description="Password for database authentication", default=environment_password),
        docs: List[Union[Document, Dict[str, Any]]] = Field(description="List of documents to update, each as a dict with 'id' (str|int), 'content' (str), optional 'metadata' (dict), 'embedding' (list[float|int]).", default=None),
        collection_name: str = Field(description="Name of the target collection.", default=None),
        ctx: Context = Field(description="FastMCP context for progress reporting", default=None),
) -> str:
    """Updates existing documents in a collection in the vector database."""
    if not docs:
        raise ValueError("docs list must not be empty")
    if not collection_name:
        raise ValueError("collection_name must not be empty")
    for doc in docs:
        if 'id' not in doc:
            raise ValueError("Each document must have 'id'")

    db = initialize_database(db_type=db_type, host=host, port=port, db_name=db_name, username=username, password=password)
    logger.debug(f"Updating {len(docs)} documents in collection: {collection_name}")

    try:
        if ctx:
            await ctx.report_progress(progress=0, total=100)
        db.update_docs(docs, collection_name)
        if ctx:
            await ctx.report_progress(progress=100, total=100)
        return f"{len(docs)} documents updated in '{collection_name}' successfully."
    except ValueError as e:
        logger.error(f"Invalid input for update_docs: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to update documents: {str(e)}")
        raise RuntimeError(f"Failed to update documents: {str(e)}")
    
def pgvector_mcp(argv):
    transport = "stdio"
    host = "0.0.0.0"
    port = 8000
    try:
        opts, args = getopt.getopt(
            argv,
            "ht:h:p:",
            ["help", "transport=", "host=", "port="],
        )
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.exit()
        elif opt in ("-t", "--transport"):
            transport = arg
        elif opt in ("-h", "--host"):
            host = arg
        elif opt in ("-p", "--port"):
            try:
                port = int(arg)
                if not (0 <= port <= 65535):
                    print(f"Error: Port {arg} is out of valid range (0-65535).")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Port {arg} is not a valid integer.")
                sys.exit(1)
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "http":
        mcp.run(transport="http", host=host, port=port)
    else:
        logger.error("Transport not supported")
        sys.exit(1)


if __name__ == "__main__":
    pgvector_mcp(sys.argv[1:])