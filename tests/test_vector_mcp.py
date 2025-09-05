import pytest
import shutil
from vector_mcp.vectordb.chromadb import (
    create_collection,
    get_collection,
    insert_docs,
    update_docs,
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture to create a temporary directory for ChromaDB storage."""
    db_path = tmp_path / "test_db"
    db_path.mkdir()
    yield str(db_path)
    if db_path.exists():
        shutil.rmtree(db_path)


# @pytest.fixture
# def chromadb_instance(temp_db_path):
#     """Fixture to initialize a ChromaDB instance."""
#     with patch("vector_mcp.vector_mcp.logger") as mock_logger:
#         db = initialize_database(
#             db_type="chromadb",
#             db_path=temp_db_path,
#             db_name="test_db",
#         )
#         assert isinstance(db, ChromaVectorDB)
#         mock_logger.error.assert_not_called()
#         yield db


@pytest.fixture
def sample_docs():
    """Fixture to provide sample documents for testing."""
    return [
        {
            "id": "1",
            "content": "Test document 1",
            "metadata": {"source": "test"},
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "id": "2",
            "content": "Test document 2",
            "metadata": {"source": "test"},
            "embedding": [0.4, 0.5, 0.6],
        },
    ]


# def test_initialize_database_invalid_type(temp_db_path):
#     """Test initializing with an invalid database type."""
#     with pytest.raises(SystemExit):
#         with patch("vector_mcp.vector_mcp.logger") as mock_logger:
#             db = initialize_database(
#                 db_type="chromadb",
#                 db_path=temp_db_path,
#                 db_name="test_db",
#             )
#             mock_logger.error.assert_called_once_with(
#                 "Failed to identify vector database from supported databases"
#             )


async def test_create_collection_success(chromadb_instance):
    """Test creating a new collection in ChromaDB."""
    result = await create_collection(
        db_type="chromadb",
        db_path=chromadb_instance.path,  # Use the path from the initialized instance
        db_name="test_db",
        collection_name="test_collection",
        overwrite=False,
        get_or_create=True,
        host=None,
        port=None,
        username=None,
        password=None,
    )
    assert "test_collection" in result
    assert "created or retrieved successfully" in result


async def test_create_collection_empty_name(chromadb_instance):
    """Test creating a collection with an empty name."""
    with pytest.raises(ValueError, match="collection_name must not be empty"):
        await create_collection(
            db_type="chromadb",
            db_path=chromadb_instance.path,
            db_name="test_db",
            collection_name="",
            host=None,
            port=None,
            username=None,
            password=None,
        )


async def test_get_collection_success(chromadb_instance):
    """Test retrieving an existing collection."""
    await create_collection(
        db_type="chromadb",
        db_path=chromadb_instance.path,
        db_name="test_db",
        collection_name="test_collection",
        get_or_create=True,
        host=None,
        port=None,
        username=None,
        password=None,
    )
    result = await get_collection(
        db_type="chromadb",
        db_path=chromadb_instance.path,
        db_name="test_db",
        collection_name="test_collection",
        host=None,
        port=None,
        username=None,
        password=None,
    )
    assert "test_collection" in result
    assert "retrieved successfully" in result


async def test_get_collection_nonexistent(chromadb_instance):
    """Test retrieving a non-existent collection."""
    with pytest.raises(RuntimeError, match="Failed to get collection"):
        await get_collection(
            db_type="chromadb",
            db_path=chromadb_instance.path,
            db_name="test_db",
            collection_name="nonexistent_collection",
            host=None,
            port=None,
            username=None,
            password=None,
        )


async def test_insert_docs_success(chromadb_instance, sample_docs):
    """Test inserting documents into a collection."""
    await create_collection(
        db_type="chromadb",
        db_path=chromadb_instance.path,
        db_name="test_db",
        collection_name="test_collection",
        get_or_create=True,
        host=None,
        port=None,
        username=None,
        password=None,
    )
    result = await insert_docs(
        db_type="chromadb",
        db_path=chromadb_instance.path,
        db_name="test_db",
        collection_name="test_collection",
        docs=sample_docs,
        upsert=False,
        host=None,
        port=None,
        username=None,
        password=None,
    )
    assert "2 documents inserted" in result
    assert "test_collection" in result


async def test_insert_docs_empty_list(chromadb_instance):
    """Test inserting an empty document list."""
    with pytest.raises(ValueError, match="docs list must not be empty"):
        await insert_docs(
            db_type="chromadb",
            db_path=chromadb_instance.path,
            db_name="test_db",
            collection_name="test_collection",
            docs=[],
            host=None,
            port=None,
            username=None,
            password=None,
        )


async def test_insert_docs_invalid_doc(chromadb_instance):
    """Test inserting documents with missing required fields."""
    invalid_docs = [{"content": "No ID document"}]
    with pytest.raises(ValueError, match="Each document must have 'id' and 'content'"):
        await insert_docs(
            db_type="chromadb",
            db_path=chromadb_instance.path,
            db_name="test_db",
            collection_name="test_collection",
            docs=invalid_docs,
            host=None,
            port=None,
            username=None,
            password=None,
        )


async def test_update_docs_success(chromadb_instance, sample_docs):
    """Test updating documents in a collection."""
    await create_collection(
        db_type="chromadb",
        db_path=chromadb_instance.path,
        db_name="test_db",
        collection_name="test_collection",
        get_or_create=True,
        host=None,
        port=None,
        username=None,
        password=None,
    )
    await insert_docs(
        db_type="chromadb",
        db_path=chromadb_instance.path,
        db_name="test_db",
        collection_name="test_collection",
        docs=sample_docs,
        upsert=False,
        host=None,
        port=None,
        username=None,
        password=None,
    )
    updated_docs = [
        {
            "id": "1",
            "content": "Updated document 1",
            "metadata": {"source": "updated"},
            "embedding": [0.7, 0.8, 0.9],
        }
    ]
    result = await update_docs(
        db_type="chromadb",
        db_path=chromadb_instance.path,
        db_name="test_db",
        collection_name="test_collection",
        docs=updated_docs,
        host=None,
        port=None,
        username=None,
        password=None,
    )
    assert "1 documents updated" in result
    assert "test_collection" in result


async def test_update_docs_empty_list(chromadb_instance):
    """Test updating with an empty document list."""
    with pytest.raises(ValueError, match="docs list must not be empty"):
        await update_docs(
            db_type="chromadb",
            db_path=chromadb_instance.path,
            db_name="test_db",
            collection_name="test_collection",
            docs=[],
            host=None,
            port=None,
            username=None,
            password=None,
        )


async def test_update_docs_invalid_doc(chromadb_instance):
    """Test updating documents with missing required fields."""
    invalid_docs = [{"content": "No ID document"}]
    with pytest.raises(ValueError, match="Each document must have 'id'"):
        await update_docs(
            db_type="chromadb",
            db_path=chromadb_instance.path,
            db_name="test_db",
            collection_name="test_collection",
            docs=invalid_docs,
            host=None,
            port=None,
            username=None,
            password=None,
        )
