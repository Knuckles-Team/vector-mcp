# Usage — MCP / API / CLI

`vector-mcp` exposes the same capability three ways: as **MCP tools** an agent calls,
as a **Python API** (`Api`) you import, and through the **agent CLI**. The complete
backend matrix and the standardized package layout are in [Overview](overview.md).

## As an MCP server

Once [deployed](deployment.md), the server registers two consolidated, action-routed
tools. Each tool dispatches on an `action` argument, which keeps the LLM tool surface
small while covering every operation.

| Tool | Toggle | Actions |
|---|---|---|
| `vector_collection_management` | `COLLECTION_MANAGEMENTTOOL` | `create_collection`, `add_documents`, `delete_collection`, `list_collections` |
| `vector_search` | `SEARCHTOOL` | `semantic_search`, `lexical_search` (BM25), `search` (hybrid) |

Example agent prompts that map onto these tools:

- *"Create a collection named `handbook` and ingest the files in `./docs`"* → `vector_collection_management`
- *"Semantically search the `handbook` collection for 'expense policy'"* → `vector_search`
- *"List every collection in the vector store"* → `vector_collection_management`

## As a Python API

`Api` (`vector_mcp.vector_api.Api`) is the client that backs the MCP tools. Build one
directly, or let `get_client()` assemble one from the environment.

```python
from vector_mcp.vector_api import Api

api = Api(
    base_url="http://localhost:8000",
    token=None,
    verify=False,
)

# Collections
api.create_collection(collection_name="handbook")
api.add_documents(collection_name="handbook", document_directory="./docs")
collections = api.list_collections()

# Retrieval
hits = api.semantic_search(collection_name="handbook", question="expense policy")
bm25 = api.lexical_search(collection_name="handbook", question="expense policy")
hybrid = api.search(collection_name="handbook", question="expense policy")
```

Build a client straight from the environment:

```python
from vector_mcp.auth import get_client
api = get_client()        # reads LLM_BASE_URL / LLM_TOKEN from the environment / .env
```

## As an agent CLI

The integrated Pydantic-AI graph agent runs as the `vector-agent` console script. It
connects to a running MCP server (`MCP_URL`) and can serve a web interface:

```bash
export MCP_URL=http://localhost:8000/mcp
export EMBEDDING_MODEL_ID=text-embedding-nomic-embed-text-v2-moe

vector-agent --provider openai --model-id gpt-4o
```

The agent speaks the Agent Control Protocol and, with `ENABLE_WEB_UI=True`, exposes a
web interface on `:9023`. See [Deployment](deployment.md#agent-server) for the
container recipe that runs the MCP server and the agent together.
