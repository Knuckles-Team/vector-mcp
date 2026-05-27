import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Save original modules
_orig_agent_utilities = sys.modules.get("agent_utilities")
_orig_llama_index = sys.modules.get("llama_index")
_orig_llama_index_core = sys.modules.get("llama_index.core")
_orig_llama_index_core_schema = sys.modules.get("llama_index.core.schema")
_orig_llama_index_core_vector_stores = sys.modules.get("llama_index.core.vector_stores")
_orig_llama_index_core_vector_stores_types = sys.modules.get("llama_index.core.vector_stores.types")

# Ensure agent_utilities is mocked or imported safely
mock_agent_utilities = MagicMock()
sys.modules["agent_utilities"] = mock_agent_utilities

# Mock LlamaIndex dependencies before importing
mock_llama_index = MagicMock()
mock_llama_index.core = MagicMock()
mock_llama_index.core.schema = MagicMock()
mock_llama_index.core.vector_stores = MagicMock()
mock_llama_index.core.vector_stores.types = MagicMock()

LlamaDocument = MagicMock()
mock_llama_index.core.schema.Document = LlamaDocument
mock_llama_index.core.SimpleDirectoryReader = MagicMock()
mock_llama_index.core.StorageContext = MagicMock()
mock_llama_index.core.VectorStoreIndex = MagicMock()
mock_llama_index.core.vector_stores.types.BasePydanticVectorStore = MagicMock()

sys.modules["llama_index"] = mock_llama_index
sys.modules["llama_index.core"] = mock_llama_index.core
sys.modules["llama_index.core.schema"] = mock_llama_index.core.schema
sys.modules["llama_index.core.vector_stores"] = mock_llama_index.core.vector_stores
sys.modules["llama_index.core.vector_stores.types"] = (
    mock_llama_index.core.vector_stores.types
)

from vector_mcp.retriever.llamaindex_retriever import LlamaIndexRetriever

# Immediately restore/clean sys.modules
for name, orig in [
    ("agent_utilities", _orig_agent_utilities),
    ("llama_index", _orig_llama_index),
    ("llama_index.core", _orig_llama_index_core),
    ("llama_index.core.schema", _orig_llama_index_core_schema),
    ("llama_index.core.vector_stores", _orig_llama_index_core_vector_stores),
    ("llama_index.core.vector_stores.types", _orig_llama_index_core_vector_stores_types),
]:
    if orig is not None:
        sys.modules[name] = orig
    else:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def mock_dependencies_fixture():
    # Set them up again during the test run
    _curr_orig = {
        name: sys.modules.get(name) for name in [
            "agent_utilities", "llama_index", "llama_index.core",
            "llama_index.core.schema", "llama_index.core.vector_stores",
            "llama_index.core.vector_stores.types"
        ]
    }
    sys.modules["agent_utilities"] = mock_agent_utilities
    sys.modules["llama_index"] = mock_llama_index
    sys.modules["llama_index.core"] = mock_llama_index.core
    sys.modules["llama_index.core.schema"] = mock_llama_index.core.schema
    sys.modules["llama_index.core.vector_stores"] = mock_llama_index.core.vector_stores
    sys.modules["llama_index.core.vector_stores.types"] = (
        mock_llama_index.core.vector_stores.types
    )
    yield
    # Restore
    for name, orig in _curr_orig.items():
        if orig is not None:
            sys.modules[name] = orig
        else:
            sys.modules.pop(name, None)



def test_llamaindex_retriever_init():
    """Verify that LlamaIndexRetriever initializes with the correct vector store."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)
    assert retriever.vector_store == mock_store
    assert retriever.file_reader_class == mock_llama_index.core.SimpleDirectoryReader


def test_llamaindex_retriever_load_doc_contents():
    """Verify loading documents directly from string contents."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    LlamaDocument.reset_mock()
    docs = retriever._load_doc(input_contents=["test content 1", "test content 2"])
    assert len(docs) == 2
    LlamaDocument.assert_any_call(text="test content 1")
    LlamaDocument.assert_any_call(text="test content 2")


def test_llamaindex_retriever_load_doc_directory_success():
    """Verify loading documents from a valid directory."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    mock_reader_class = MagicMock()
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = ["doc1", "doc2"]
    mock_reader_class.return_value = mock_reader

    retriever.file_reader_class = mock_reader_class

    with patch("os.path.exists", return_value=True):
        docs = retriever._load_doc(input_dir="/some/dir")
        assert docs == ["doc1", "doc2"]
        mock_reader_class.assert_called_once_with(input_dir="/some/dir")
        mock_reader.load_data.assert_called_once()


def test_llamaindex_retriever_load_doc_directory_not_found():
    """Verify error raised when document directory is not found."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    with patch("os.path.exists", return_value=False):
        with pytest.raises(ValueError, match="Input directory not found"):
            retriever._load_doc(input_dir="/some/dir")


def test_llamaindex_retriever_load_doc_paths_success():
    """Verify loading documents from a list of paths."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    mock_reader_class = MagicMock()
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = ["doc3"]
    mock_reader_class.return_value = mock_reader

    retriever.file_reader_class = mock_reader_class

    with patch("os.path.exists", return_value=True):
        docs = retriever._load_doc(input_docs=["/some/file.txt"])
        assert docs == ["doc3"]
        mock_reader_class.assert_called_once_with(input_files=["/some/file.txt"])
        mock_reader.load_data.assert_called_once()


def test_llamaindex_retriever_load_doc_paths_not_found():
    """Verify error raised when document file is not found."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    with patch("os.path.exists", return_value=False):
        with pytest.raises(ValueError, match="Document file not found"):
            retriever._load_doc(input_docs=["/some/file.txt"])


def test_llamaindex_retriever_load_doc_empty():
    """Verify error raised when no loading sources are provided."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)
    with pytest.raises(
        ValueError, match="No input directory, docs, or content provided"
    ):
        retriever._load_doc()


def test_llamaindex_retriever_initialize_collection():
    """Verify correct StorageContext and VectorStoreIndex initialization from documents."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    mock_storage_context = MagicMock()
    mock_llama_index.core.StorageContext.from_defaults.return_value = (
        mock_storage_context
    )

    mock_index = MagicMock()
    mock_llama_index.core.VectorStoreIndex.from_documents.return_value = mock_index

    with patch.object(retriever, "_load_doc", return_value=["doc1", "doc2"]):
        res = retriever.initialize_collection(document_contents=["test"])
        assert res is True
        assert retriever.storage_context == mock_storage_context
        assert retriever.index == mock_index
        mock_llama_index.core.StorageContext.from_defaults.assert_called_with(
            vector_store=mock_store
        )
        mock_llama_index.core.VectorStoreIndex.from_documents.assert_called_with(
            documents=["doc1", "doc2"],
            storage_context=mock_storage_context,
            embed_model=retriever.embed_model,
        )


def test_llamaindex_retriever_connect_database():
    """Verify correct StorageContext and VectorStoreIndex initialization from vector store."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    mock_storage_context = MagicMock()
    mock_llama_index.core.StorageContext.from_defaults.return_value = (
        mock_storage_context
    )

    mock_index = MagicMock()
    mock_llama_index.core.VectorStoreIndex.from_vector_store.return_value = mock_index

    res = retriever.connect_database()
    assert res is True
    assert retriever.index == mock_index
    mock_llama_index.core.VectorStoreIndex.from_vector_store.assert_called_with(
        vector_store=mock_store,
        storage_context=mock_storage_context,
        embed_model=retriever.embed_model,
    )


def test_llamaindex_retriever_add_documents():
    """Verify that adding documents inserts them into the active index."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    mock_index = MagicMock()
    retriever.index = mock_index

    with patch.object(retriever, "_load_doc", return_value=["new_doc"]):
        docs = retriever.add_documents(document_contents=["new"])
        assert docs == ["new_doc"]
        mock_index.insert.assert_called_once_with("new_doc")


def test_llamaindex_retriever_query():
    """Verify query mapping and retriever execution logic."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    mock_index = MagicMock()
    mock_retriever_obj = MagicMock()

    node1 = MagicMock()
    node1.node.get_content.return_value = "content1"
    node1.node.node_id = "id1"
    node1.node.metadata = {"meta1": "val1"}
    node1.score = 0.9

    mock_retriever_obj.retrieve.return_value = [node1]
    mock_index.as_retriever.return_value = mock_retriever_obj
    retriever.index = mock_index

    results = retriever.query(question="how to test?", number_results=5)
    assert len(results) == 1
    assert results[0] == {
        "text": "content1",
        "score": 0.9,
        "id": "id1",
        "metadata": {"meta1": "val1"},
    }
    mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)
    mock_retriever_obj.retrieve.assert_called_once_with(
        str_or_query_bundle="how to test?"
    )


def test_llamaindex_retriever_bm25_query():
    """Verify BM25 query fallback execution."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    with patch.object(retriever, "query", return_value=["result"]) as mock_query:
        res = retriever.bm25_query("test question", 3, extra_arg="val")
        assert res == ["result"]
        mock_query.assert_called_once_with("test question", 3, extra_arg="val")


def test_llamaindex_retriever_validate_index_missing():
    """Verify exception is raised when index is not initialized."""
    mock_store = MagicMock()
    retriever = LlamaIndexRetriever(vector_store=mock_store)

    with pytest.raises(Exception, match="Query index is not initialized"):
        retriever.query("test", 1)
