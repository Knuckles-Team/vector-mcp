
import sys

# Mock missing dependencies
sys.modules['llama_index.embeddings.huggingface'] = None
sys.modules['llama_index.vector_stores.postgres'] = None
sys.modules['qdrant_client'] = None
sys.modules['llama_index.vector_stores.qdrant'] = None
# Mock others if needed

print("Checking vector_mcp.vector_mcp import...")
try:
    import vector_mcp.vector_mcp
    print("Success: vector_mcp.vector_mcp imported.")
except Exception as e:
    print(f"Failed: vector_mcp.vector_mcp import raised {e}")
    sys.exit(1)

print("Checking vector_mcp.retriever import...")
try:
    import vector_mcp.retriever
    print("Success: vector_mcp.retriever imported.")
except Exception as e:
    print(f"Failed: vector_mcp.retriever import raised {e}")
    sys.exit(1)

print("All chain checks passed.")
