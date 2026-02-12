import requests
import time

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

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "list_collections", "arguments": {}},
    }

    try:

        endpoint = f"http://localhost:{port}/mcp"

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(
            f"http://localhost:{port}/mcp", json=payload, headers=headers, timeout=10
        )

        if response.status_code == 200:
            print("  [PASS] Connection established. Status: 200")
            print(f"  Response: {response.text[:100]}...")

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
            print("  [WARN] 404 at /messages. Trying /sse...")
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
