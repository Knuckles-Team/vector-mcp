import os
import sys
from llama_index.core import Settings

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from vector_mcp.vector_mcp import config

    print(f"Current Config: {config}")
except ImportError:
    print("Could not import vector_mcp")

print(f"Default Global Chunk Size: {Settings.chunk_size}")

if Settings.chunk_size > 512:
    print("FAILURE: Chunk size is larger than 512 tokens.")
else:
    print("SUCCESS: Chunk size is 512 tokens or less.")
