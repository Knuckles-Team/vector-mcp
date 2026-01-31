
import sys
import os

print(f"Python: {sys.version}")

try:
    import psycopg2
    print("SUCCESS: psycopg2 imported")
except ImportError as e:
    print(f"FAILURE: psycopg2 not found: {e}")

try:
    import psycopg
    print("SUCCESS: psycopg (v3) imported")
except ImportError as e:
    print(f"FAILURE: psycopg not found: {e}")

try:
    from llama_index.vector_stores.postgres import PGVectorStore
    print("SUCCESS: PGVectorStore imported")
except ImportError as e:
    print(f"FAILURE: PGVectorStore import failed: {e}")

if "psycopg" in sys.modules:
    # Try to connect using psycopg directly
    print("\nAttempting raw psycopg connection...")
    try:
        conn = psycopg.connect(
            host="postgres",
            port="5432",
            user="postgres",
            password="password",
            dbname="vectordb",
            connect_timeout=5
        )
        print("SUCCESS: Raw psycopg connection established")
        conn.close()
    except Exception as e:
        print(f"FAILURE: Raw psycopg connection failed: {e}")

# Try PGVectorStore
print("\nAttempting PGVectorStore init...")
try:
    # Try forcing asyncpg if needed, or psycopg
    store = PGVectorStore.from_params(
        host="postgres",
        port="5432",
        user="postgres",
        password="password",
        database="vectordb",
        table_name="test_collection",
        embed_dim=1536
    )
    print("SUCCESS: PGVectorStore initialized")
except Exception as e:
    print(f"FAILURE: PGVectorStore init failed: {e}")
