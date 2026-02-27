import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

print(f"Python: {sys.version}")

try:
    from vector_mcp.vectordb.postgres import PostgreSQL
    from agent_utilities.embedding_utilities import create_embedding_model

    print("SUCCESS: Imported PostgreSQL")
except ImportError as e:
    print(f"FAILURE: Could not import PostgreSQL: {e}")
    sys.exit(1)

print("\n--- Testing create_embedding_model ---")
try:
    embed_model = create_embedding_model()
    print(f"Got embedding model: {embed_model}")
    print("Generating test embedding...")
    emb = embed_model.get_text_embedding("test")
    print(f"Embedding generated. Dim: {len(emb)}")
except Exception as e:
    print(f"FAILURE: Embedding model check failed: {e}")

print("\n--- Testing PostgreSQL init ---")
try:
    print("Initializing PostgreSQL...")
    db = PostgreSQL(
        host="postgres",
        port="5432",
        dbname="vectordb",
        username="postgres",
        password="password",
        collection_name="memory",
    )
    print("SUCCESS: PostgreSQL initialized")

    print("Creating collection...")
    db.create_collection("memory", overwrite=False)
    print("SUCCESS: Collection created")

    print("Getting index...")
    idx = db._get_index()
    print("SUCCESS: Index retrieved")

except Exception as e:
    print(f"FAILURE: PostgreSQL init failed: {e}")
    import traceback

    traceback.print_exc()
