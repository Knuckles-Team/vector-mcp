import requests
import time

# Configuration
SERVERS = [
    {"name": "ChromaDB", "port": 8003},
    {"name": "PGVector", "port": 8004},
    {"name": "MongoDB", "port": 8005},
    {"name": "Couchbase", "port": 8006},
    {"name": "Qdrant", "port": 8007},
]

BASE_URL = "http://localhost"


def test_server(server_config):
    name = server_config["name"]
    port = server_config["port"]
    url = f"{BASE_URL}:{port}/mcp"

    print(f"Testing {name} on port {port}...")

    # 1. Initialization / SSE check (or just list tools if using HTTP)
    # The server uses "streamable-http" or "sse". If it's FastMCP, /sse might be the endpoint, or we can use the messages endpoint.
    # The `vector-mcp` code seems to use FastMCP. FastMCP usually exposes /sse and /messages.
    # Let's try to query the tools list via JSON-RPC over HTTP if supported, or just hit /sse to see if it connects.
    # For simplicity, if we don't have a full MCP client, sending a POST to /messages works for standard MCP.

    # Actually, let's look at `server.py` arguments again. It supports `streamable-http`.
    # This usually means we can POST to `/messages`.

    # List Collections
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "list_collections", "arguments": {}},
    }

    try:
        # Note: The endpoint might differ depending on FastMCP version.
        # Usually it's /mcp/messages or just /messages.
        # The compose has `PORT=800X`.
        # `fastmcp` usually mounts at `/sse` and `/messages`.
        # Let's try `/messages`.

        # NOTE: Config says `TRANSPORT=streamable-http`.
        # FastMCP with streamable-http usually listens on the root or configured path.
        # But let's assume standard MCP HTTP binding.

        endpoint = f"http://localhost:{port}/mcp"
        # However, checking `vector-mcp` code, it uses `FastMCP`.
        # If it uses starlette/fastapi under the hood, the default might be at root or /messages.

        # Let's fail fast: ping the health or root
        # Attempt listing collections
        # NOTE: The server logs show it listens on /mcp
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(
            f"http://localhost:{port}/mcp", json=payload, headers=headers, timeout=10
        )

        if response.status_code == 200:
            print("  [PASS] Connection established. Status: 200")
            print(f"  Response: {response.text[:100]}...")

            # Create Collection
            create_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "create_collection",
                    "arguments": {
                        "collection_name": f"test_collection_{int(time.time())}",
                        "get_or_create": True,
                    },
                },
            }
            resp_create = requests.post(
                f"http://localhost:{port}/mcp",
                json=create_payload,
                headers=headers,
                timeout=30,
            )
            if resp_create.status_code == 200 and "error" not in resp_create.json():
                print("  [PASS] create_collection success.")
            else:
                print(f"  [FAIL] create_collection failed: {resp_create.text}")

        elif response.status_code == 404:
            # Try /mcp/messages if mapped there (unlikely based on code but possible)
            print("  [WARN] 404 at /messages. Trying /sse...")
            # Just checking if we can connect to /sse
            sse_resp = requests.get(
                f"http://localhost:{port}/sse", stream=True, timeout=5
            )
            if sse_resp.status_code == 200:
                print("  [PASS] SSE endpoint exists.")
            else:
                print("  [FAIL] Could not connect to MCP server.")
        else:
            print(f"  [FAIL] Server returned {response.status_code}")

    except Exception as e:
        print(f"  [FAIL] Exception: {e}")


def main():
    print("Waiting for services to settle (10s)...")
    time.sleep(10)

    for server in SERVERS:
        test_server(server)
        print("-" * 30)


if __name__ == "__main__":
    main()
