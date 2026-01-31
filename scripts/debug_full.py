
import sys
import os
import logging

# Configure logging to see LlamaIndex logs
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

print(f"Python: {sys.version}")

try:
    from vector_mcp.vectordb.pgvector import PGVectorDB
    from vector_mcp.utils import get_embedding_model
    print("SUCCESS: Imported PGVectorDB")
except ImportError as e:
    print(f"FAILURE: Could not import PGVectorDB: {e}")
    sys.exit(1)

print("\n--- Testing get_embedding_model ---")
try:
    embed_model = get_embedding_model()
    print(f"Got embedding model: {embed_model}")
    print("Generating test embedding...")
    emb = embed_model.get_text_embedding("test")
    print(f"Embedding generated. Dim: {len(emb)}")
except Exception as e:
    print(f"FAILURE: Embedding model check failed: {e}")
    # Don't exit, try DB anyway if possible (but DB relies on it)

print("\n--- Testing PGVectorDB init ---")
try:
    print("Initializing PGVectorDB...")
    # Use params from logs
    db = PGVectorDB(
        host="postgres",
        port="5432",
        dbname="vectordb",
        username="postgres",
        password="password",
        collection_name="memory"
    )
    print("SUCCESS: PGVectorDB initialized")
    
    print("Creating collection...")
    db.create_collection("memory", overwrite=False)
    print("SUCCESS: Collection created")

    print("Getting index...")
    idx = db._get_index()
    print("SUCCESS: Index retrieved")

except Exception as e:
    print(f"FAILURE: PGVectorDB init failed: {e}")
    import traceback
    traceback.print_exc()
