
import os
import sys
from openai import OpenAI

print(f"Python: {sys.version}")

base_url = os.environ.get("OPENAI_BASE_URL", "http://host.docker.internal:1234/v1")
api_key = os.environ.get("OPENAI_API_KEY", "llama")

print(f"Testing OpenAI connection to: {base_url}")

try:
    client = OpenAI(base_url=base_url, api_key=api_key)
    # List models or generate embedding
    print("Attempting to list models...")
    models = client.models.list()
    print("SUCCESS: Listed models")
    for m in models:
        print(f" - {m.id}")

    print("\nAttempting to generate embedding...")
    resp = client.embeddings.create(
        input="test",
        model="text-embedding-nomic-embed-text-v1.5" # Default in vector_mcp.py
    )
    print("SUCCESS: Generated embedding")
    print(f"Dimension: {len(resp.data[0].embedding)}")

except Exception as e:
    print(f"FAILURE: OpenAI connection failed: {e}")
