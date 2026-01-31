
import os
import sys
from typing import Any

# Mock env vars to avoid errors during instantiation if keys are missing
os.environ["OPENAI_API_KEY"] = "sk-test-key" 

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    print("Successfully imported OpenAIEmbedding")
except ImportError as e:
    print(f"Failed to import OpenAIEmbedding: {e}")
    OpenAIEmbedding = None

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
    print("Successfully imported OllamaEmbedding")
except ImportError as e:
    print(f"Failed to import OllamaEmbedding: {e}")
    OllamaEmbedding = None

def inspect_openai():
    if not OpenAIEmbedding:
        return
    
    print("\n--- Inspecting OpenAIEmbedding ---")
    try:
        # Initialize with timeout
        embed_model = OpenAIEmbedding(model="text-embedding-3-small", timeout=32400.0)
        print(f"Initialized OpenAIEmbedding with timeout=32400.0")
        
        # Force client initialization usually happens on first call or via _get_client
        # We can try to access the property .client or call a private method if needed, 
        # or just run a dummy embedding if credentials allowed (might fail if key is invalid)
        
        if hasattr(embed_model, "_get_client"):
            print("Calling _get_client()")
            client = embed_model._get_client()
            print(f"Client type: {type(client)}")
            if hasattr(client, "timeout"):
                print(f"Client.timeout: {client.timeout}")
            else:
                print("Client does not have 'timeout' attribute")

        # Test explicit http_client injection
        try:
            import httpx
            custom_client = httpx.Client(timeout=32400.0)
            print("\nTesting explicit http_client injection...")
            embed_model_custom = OpenAIEmbedding(model="text-embedding-3-small", http_client=custom_client)
            
            # Check if client uses our custom client or configures one with same timeout
            if hasattr(embed_model_custom, "_get_client"):
                client_c = embed_model_custom._get_client()
                # openai client _client is the httpx client usually, or it wraps it
                if hasattr(client_c, "_client"): # OpenAI Python Client stores httpx client in _client
                     print(f"OpenAI Internal HTTPX Client Timeout: {client_c._client.timeout}")
                elif hasattr(client_c, "timeout"):
                     print(f"OpenAI Client Wrapper Timeout: {client_c.timeout}")
                
        except Exception as e:
            print(f"Error testing http_client injection: {e}")
                
        # Check standard openai client attributes if possible
        if hasattr(embed_model, "_client") and embed_model._client:
             print(f"_client.timeout: {embed_model._client.timeout}")

    except Exception as e:
        print(f"Error inspecting OpenAIEmbedding: {e}")

def inspect_ollama():
    if not OllamaEmbedding:
        return

    print("\n--- Inspecting OllamaEmbedding ---")
    try:
        # Ollama might not take timeout in init in some versions, or might pass it to client
        embed_model = OllamaEmbedding(model_name="nomic-embed-text", timeout=32400.0)
        print(f"Initialized OllamaEmbedding with timeout=32400.0")
        
        # Check attributes - Ollama usually uses simple httpx calls or client
        print(f"OllamaEmbedding instance attributes: {embed_model.__dict__.keys()}")
        
        # Often stored in _client or similar if using base class
        if hasattr(embed_model, "_client"):
             print(f"Found _client: {embed_model._client}")
             if hasattr(embed_model._client, "timeout"):
                 print(f"_client.timeout: {embed_model._client.timeout}")

    except TypeError as e:
        print(f"TypeError initializing OllamaEmbedding: {e}")
        # Try without timeout to see what it has
        try:
            embed_model = OllamaEmbedding(model_name="nomic-embed-text")
            print("Initialized OllamaEmbedding WITHOUT timeout")
            print(f"OllamaEmbedding instance attributes: {embed_model.__dict__.keys()}")
        except Exception as e2:
            print(f"Error initializing OllamaEmbedding without timeout: {e2}")

    except Exception as e:
        print(f"Error inspecting OllamaEmbedding: {e}")

if __name__ == "__main__":
    inspect_openai()
    inspect_ollama()
