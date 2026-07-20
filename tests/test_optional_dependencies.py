import sys
from importlib import import_module


def test_optional_dependencies():
    # Save original sys.modules state of optional dependencies
    keys_to_restore = [
        "psycopg_pool",
        "qdrant_client",
        "pymongo",
    ]
    orig_modules = {k: sys.modules.get(k) for k in keys_to_restore}

    # Save original vector_mcp modules state so we can restore it exactly
    orig_vector_mcp = {
        k: sys.modules[k]
        for k in list(sys.modules.keys())
        if k.startswith("vector_mcp")
    }

    try:
        # Mock missing optional dependencies
        for k in keys_to_restore:
            sys.modules[k] = None  # type: ignore

        # Unload vector_mcp modules so they are re-imported under mock conditions
        for k in orig_vector_mcp:
            sys.modules.pop(k, None)

        # Import target modules
        import_module("vector_mcp")
        import_module("vector_mcp.vectordb")

    finally:
        # Restore original sys.modules state of optional dependencies
        for k, v in orig_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

        # Unload any new vector_mcp modules imported under mock conditions
        for k in list(sys.modules.keys()):
            if k.startswith("vector_mcp") and k not in orig_vector_mcp:
                sys.modules.pop(k, None)

        # Restore original vector_mcp modules
        for k, v in orig_vector_mcp.items():
            sys.modules[k] = v
