import pytest
import shutil
from unittest.mock import MagicMock, patch
from vector_mcp.vectordb.chromadb import ChromaVectorDB
from vector_mcp.vectordb.base import Document


@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture to create a temporary directory for ChromaDB storage."""
    db_path = tmp_path / "test_db"
    db_path.mkdir()
    yield str(db_path)
    if db_path.exists():
        shutil.rmtree(db_path)


@pytest.fixture
def chromadb_instance(temp_db_path):
    """Fixture to initialize a ChromaDB instance."""
    # We mock get_embedding_model to avoid model loading overhead/errors in test
    with patch("vector_mcp.vectordb.chromadb.get_embedding_model") as mock_embed:
        mock_embed.return_value = MagicMock()
        mock_embed.return_value.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        db = ChromaVectorDB(path=temp_db_path, collection_name="test_collection")
        yield db


@pytest.fixture
def sample_docs():
    """Fixture to provide sample documents for testing."""
    return [
        {
            "id": "1",
            "content": "Test document 1",
            "metadata": {"source": "test"},
            # "embedding": [0.1, 0.2, 0.3], # LlamaIndex might handle embedding if not provided
        },
        {
            "id": "2",
            "content": "Test document 2",
            "metadata": {"source": "test"},
            # "embedding": [0.4, 0.5, 0.6],
        },
    ]


def test_create_collection_success(chromadb_instance):
    """Test creating a new collection in ChromaDB."""
    # For ChromaDB, create_collection wraps get_or_create_collection
    collection = chromadb_instance.create_collection(
        collection_name="test_collection_new", overwrite=False
    )
    assert collection is not None
    # Verify it exists in list
    colls = chromadb_instance.get_collections()
    # Chroma client list_collections returns list of Collection objects or names depending on version?
    # Usually Collection objects.
    names = [c.name for c in colls]
    assert "test_collection_new" in names


def test_get_collection_success(chromadb_instance):
    """Test retrieving an existing collection."""
    chromadb_instance.create_collection("test_collection_2")
    coll = chromadb_instance.get_collection("test_collection_2")
    assert coll.name == "test_collection_2"


def test_insert_docs_success(chromadb_instance, sample_docs):
    """Test inserting documents into a collection."""
    # Mocking internal get_index to avoid LlamaIndex complex setup if possible
    # But ChromaVectorDB uses LlamaIndex VectorStore index.insert_documents

    # We really want to verify integration with LlamaIndex logic, but that requires embedding model
    # which we mocked.

    # If we want to test insert_documents, we should mock the _get_index method or the vector_store
    with patch.object(chromadb_instance, "_get_index") as mock_get_index:
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index

        chromadb_instance.insert_documents(sample_docs)

        mock_index.insert_documents.assert_called_once()
        # Verify arguments passed to insert_documents are LIDocuments
        call_args = mock_index.insert_documents.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].text == "Test document 1"


def test_update_docs_success(chromadb_instance, sample_docs):
    """Test updating documents in a collection."""
    with patch.object(chromadb_instance, "_get_index") as mock_get_index:
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index

        chromadb_instance.update_documents(sample_docs)

        # update just calls insert with upsert? Logic in code: insert_documents(upsert=True)
        # Check logic inside insert_documents... it just calls index.insert_documents.
        # LlamaIndex insert_documents behavior depends on vector store.
        mock_index.insert_documents.assert_called_once()
