import sys
from importlib import import_module


def _snapshot(names):
    return {k: sys.modules.get(k) for k in names}


def _restore_modules(snapshot):
    for k, v in snapshot.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _unload_prefixed(prefix, keep):
    for k in list(sys.modules.keys()):
        if k.startswith(prefix) and k not in keep:
            sys.modules.pop(k, None)


def test_optional_dependencies():
    # Save original sys.modules state of optional dependencies
    keys_to_restore = [
        "psycopg_pool",
        "qdrant_client",
        "pymongo",
    ]
    orig_modules = _snapshot(keys_to_restore)

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
        _unload_prefixed("vector_mcp", orig_vector_mcp)

        # Import target modules
        import_module("vector_mcp")
        import_module("vector_mcp.vectordb")

    finally:
        # Restore original sys.modules state of optional dependencies
        _restore_modules(orig_modules)

        # Unload any new vector_mcp modules imported under mock conditions
        _unload_prefixed("vector_mcp", orig_vector_mcp)

        # Restore original vector_mcp modules
        for k, v in orig_vector_mcp.items():
            sys.modules[k] = v
