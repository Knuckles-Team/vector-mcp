import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.getcwd())

from vector_mcp.retriever.chromadb_retriever import ChromaDBRetriever

def test_embedding():
    # Set environment variables for local testing
    os.environ["EMBEDDING_PROVIDER"] = "openai"
    # Assuming running locally on the host, so localhost is appropriate.
    # If running in a container, this might need to be host.docker.internal or correct IP
    os.environ["OPENAI_BASE_URL"] = "http://localhost:1234/v1" 
    os.environ["OPENAI_API_KEY"] = "llama"
    os.environ["EMBEDDING_MODEL"] = "text-embedding-nomic-embed-text-v1.5"
    
    # Check if documents directory exists
    docs_path = Path("./mcp/documents")
    if not docs_path.exists():
        print(f"Creating {docs_path}...")
        docs_path.mkdir(parents=True, exist_ok=True)
        # Create a dummy file if empty
        with open(docs_path / "test.txt", "w") as f:
            f.write("This is a test document for embedding validation.")

    print("Initializing ChromaDBRetriever...")
    # Use a local path for the database to avoid connection issues during test
    db_path = "./test_chroma_db"
    
    retriever = ChromaDBRetriever(
        path=db_path,
        collection_name="test_validation"
    )

    print(f"Ingesting documents from {docs_path}...")
    try:
        success = retriever.initialize_collection(
            document_directory=str(docs_path),
            overwrite=True
        )
        
        if success:
            print("Verifying embeddings by querying...")
            # Simple query to verify
            results = retriever.query(question="test document", number_results=1)
            print(f"Query Result: {results}")
            print("\nSUCCESS: Embeddings generated and queried successfully.")
        else:
            print("\nFAILURE: initialize_collection returned False.")

    except Exception as e:
        print(f"\nFAILURE: Exception during embedding: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embedding()
