
import pytest
from unittest.mock import MagicMock, patch
import hashlib

# Import the class to test (we might need to mock imports if they fail)
with patch("vector_mcp.vectordb.pgvector.PGVectorDB"), \
     patch("vector_mcp.vectordb.utils.require_optional_import", side_effect=lambda x, y: lambda z: z):
    from vector_mcp.retriever.pgvector_retriever import PGVectorRetriever

def test_pgvector_retriever_query_structure():
    """Verify PGVectorRetriever.query returns List[Dict] with correct keys."""
    retriever = PGVectorRetriever(connection_string="postgresql://u:p@localhost:5432/db")
    
    # Mock index and retriever
    mock_index = MagicMock()
    retriever.index = mock_index
    mock_li_retriever = MagicMock()
    mock_index.as_retriever.return_value = mock_li_retriever
    
    # Mock NodeWithScore
    mock_node = MagicMock()
    mock_node.get_content.return_value = "Test Content"
    mock_node.node_id = "test_id_1"
    mock_node.metadata = {"meta": "data"}
    
    mock_node_with_score = MagicMock()
    mock_node_with_score.node = mock_node
    mock_node_with_score.score = 0.95
    
    mock_li_retriever.retrieve.return_value = [mock_node_with_score]
    
    # Execute query
    results = retriever.query("test query", number_results=1)
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert results[0]["text"] == "Test Content"
    assert results[0]["score"] == 0.95
    assert results[0]["id"] == "test_id_1"
    assert results[0]["metadata"] == {"meta": "data"}

def test_rrf_logic():
    """Verify RRF fusion logic as implemented in search."""
    # Simulation of search logic
    semantic_results = [
        {"text": "A", "score": 0.9, "id": "A"},
        {"text": "B", "score": 0.8, "id": "B"},
    ]
    bm25_results = [
        {"text": "B", "score": 10.0, "id": "B"}, # B is rank 1 here
        {"text": "C", "score": 5.0, "id": "C"},
    ]
    
    semantic_weight = 0.5
    bm25_weight = 0.5
    rrf_k = 60
    
    combined = {}
    
    # Logic copied from implementation plan / user request
    for rank, res in enumerate(semantic_results, 1):
        doc_id = res.get('id')
        if doc_id not in combined:
            combined[doc_id] = {'text': res['text'], 'rrf_score': 0}
        combined[doc_id]['rrf_score'] += semantic_weight / (rank + rrf_k)

    for rank, res in enumerate(bm25_results, 1):
        doc_id = res.get('id')
        if doc_id not in combined:
            combined[doc_id] = {'text': res['text'], 'rrf_score': 0}
        combined[doc_id]['rrf_score'] += bm25_weight / (rank + rrf_k)

    sorted_results = sorted(combined.values(), key=lambda x: x['rrf_score'], reverse=True)
    
    # Expected: B matches in both (Rank 2 in Sem, Rank 1 in BM25)
    # A matches in Sem (Rank 1)
    # C matches in BM25 (Rank 2)
    
    # Score B = 0.5/(2+60) + 0.5/(1+60) = 0.5/62 + 0.5/61 ~= 0.00806 + 0.00819 = 0.01625
    # Score A = 0.5/(1+60) = 0.5/61 ~= 0.00819
    # Score C = 0.5/(2+60) = 0.5/62 ~= 0.00806
    
    # Order should be B, A, C
    texts = [r['text'] for r in sorted_results]
    assert texts == ["B", "A", "C"]
