# Vector Database MCP Server

![PyPI - Version](https://img.shields.io/pypi/v/vector-mcp)
![PyPI - Downloads](https://img.shields.io/pypi/dd/vector-mcp)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/vector-mcp)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/vector-mcp)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/vector-mcp)
![PyPI - License](https://img.shields.io/pypi/l/vector-mcp)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/vector-mcp)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/vector-mcp)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/vector-mcp)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/vector-mcp)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/vector-mcp)

![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/vector-mcp)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/vector-mcp)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/vector-mcp)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/vector-mcp)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/vector-mcp)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/vector-mcp)

*Version: 0.1.10*

This is an MCP Server implementation which allows for a standardized
collection management system across vector database technologies.

This was heavily inspired by the RAG implementation of Microsoft's Autogen V1 framework, however,
this was changed to an MCP server model instead.

AI Agents can:

- Create collections with documents stored on the local filesystem or URLs
- Add documents to a collection
- Utilize collection for retrieval augmented generation (RAG)
- Delete collection

Supports:

- ChromaDB
- PGVector - 90% Tested
- Couchbase - 80% Tested
- Qdrant - 80% Tested
- MongoDB - 80% Tested

This repository is actively maintained - Contributions and bug reports are welcome!

Automated tests are planned

<details>
  <summary><b>Usage:</b></summary>


### MCP CLI

| Short Flag | Long Flag                          | Description                                                                 |
|------------|------------------------------------|-----------------------------------------------------------------------------|
| -h         | --help                             | Display help information                                                    |
| -t         | --transport                        | Transport method: 'stdio', 'http', or 'sse' [legacy] (default: stdio)       |
| -s         | --host                             | Host address for HTTP transport (default: 0.0.0.0)                          |
| -p         | --port                             | Port number for HTTP transport (default: 8000)                              |
|            | --auth-type                        | Authentication type: 'none', 'static', 'jwt', 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (default: none) |
|            | --token-jwks-uri                   | JWKS URI for JWT verification                                              |
|            | --token-issuer                     | Issuer for JWT verification                                                |
|            | --token-audience                   | Audience for JWT verification                                              |
|            | --oauth-upstream-auth-endpoint     | Upstream authorization endpoint for OAuth Proxy                             |
|            | --oauth-upstream-token-endpoint    | Upstream token endpoint for OAuth Proxy                                    |
|            | --oauth-upstream-client-id         | Upstream client ID for OAuth Proxy                                         |
|            | --oauth-upstream-client-secret     | Upstream client secret for OAuth Proxy                                     |
|            | --oauth-base-url                   | Base URL for OAuth Proxy                                                   |
|            | --oidc-config-url                  | OIDC configuration URL                                                     |
|            | --oidc-client-id                   | OIDC client ID                                                             |
|            | --oidc-client-secret               | OIDC client secret                                                         |
|            | --oidc-base-url                    | Base URL for OIDC Proxy                                                    |
|            | --remote-auth-servers              | Comma-separated list of authorization servers for Remote OAuth             |
|            | --remote-base-url                  | Base URL for Remote OAuth                                                  |
|            | --allowed-client-redirect-uris     | Comma-separated list of allowed client redirect URIs                       |
|            | --eunomia-type                     | Eunomia authorization type: 'none', 'embedded', 'remote' (default: none)   |
|            | --eunomia-policy-file              | Policy file for embedded Eunomia (default: mcp_policies.json)              |
|            | --eunomia-remote-url               | URL for remote Eunomia server                                              |

### Using as an MCP Server

The MCP Server can be run in two modes: `stdio` (for local testing) or `http` (for networked access). To start the server, use the following commands:

#### Run in stdio mode (default):
```bash
vector-mcp --transport "stdio"
```

#### Run in HTTP mode:
```bash
vector-mcp --transport "http"  --host "0.0.0.0"  --port "8000"
```

### Creating Collection
AI Prompt:
```text
Create a collection called zapdos with the documents in this directory:
/home/user/Documents/Chroma
```

AI Response:
```text
The collection named "zapdos" has been successfully created or retrieved from the vector database
using the documents in the directory /home/user/Documents/Chroma.
Let me know if you'd like to perform any further actions, such as querying the collection or adding more documents!
```

### Retrieving from Collection

AI Prompt:
```text
Create a collection called zapdos with the documents in this directory:
/home/user/Documents/Chroma
```

AI Response:
```text
The collection named "zapdos" has been successfully created or retrieved from the vector database
using the documents in the directory /home/user/Documents/Chroma.
Let me know if you'd like to perform any further actions, such as querying the collection or adding more documents!
```

### Deleting Collection

AI Prompt:
```text
Delete the collection called memory
```

AI Response:
```text
The collection named "memory" has been successfully deleted.
Let me know if you'd like to create a new collection or perform any other actions!
```

### Deploy MCP Server as a Service

The MCP server can be deployed using Docker, with configurable authentication, middleware, and Eunomia authorization.

#### Using Docker Run

```bash
docker pull knucklessg1/vector-mcp:latest

docker run -d \
  --name vector-mcp \
  -p 8004:8004 \
  -e HOST=0.0.0.0 \
  -e PORT=8004 \
  -e TRANSPORT=http \
  -e AUTH_TYPE=none \
  -e EUNOMIA_TYPE=none \
  knucklessg1/vector-mcp:latest
```

For advanced authentication (e.g., JWT, OAuth Proxy, OIDC Proxy, Remote OAuth) or Eunomia, add the relevant environment variables:

```bash
docker run -d \
  --name vector-mcp \
  -p 8004:8004 \
  -e HOST=0.0.0.0 \
  -e PORT=8004 \
  -e TRANSPORT=http \
  -e AUTH_TYPE=oidc-proxy \
  -e OIDC_CONFIG_URL=https://provider.com/.well-known/openid-configuration \
  -e OIDC_CLIENT_ID=your-client-id \
  -e OIDC_CLIENT_SECRET=your-client-secret \
  -e OIDC_BASE_URL=https://your-server.com \
  -e ALLOWED_CLIENT_REDIRECT_URIS=http://localhost:*,https://*.example.com/* \
  -e EUNOMIA_TYPE=embedded \
  -e EUNOMIA_POLICY_FILE=/app/mcp_policies.json \
  knucklessg1/vector-mcp:latest
```

#### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
services:
  vector-mcp:
    image: knucklessg1/vector-mcp:latest
    environment:
      - HOST=0.0.0.0
      - PORT=8004
      - TRANSPORT=http
      - AUTH_TYPE=none
      - EUNOMIA_TYPE=none
    ports:
      - 8004:8004
```

For advanced setups with authentication and Eunomia:

```yaml
services:
  vector-mcp:
    image: knucklessg1/vector-mcp:latest
    environment:
      - HOST=0.0.0.0
      - PORT=8004
      - TRANSPORT=http
      - AUTH_TYPE=oidc-proxy
      - OIDC_CONFIG_URL=https://provider.com/.well-known/openid-configuration
      - OIDC_CLIENT_ID=your-client-id
      - OIDC_CLIENT_SECRET=your-client-secret
      - OIDC_BASE_URL=https://your-server.com
      - ALLOWED_CLIENT_REDIRECT_URIS=http://localhost:*,https://*.example.com/*
      - EUNOMIA_TYPE=embedded
      - EUNOMIA_POLICY_FILE=/app/mcp_policies.json
    ports:
      - 8004:8004
    volumes:
      - ./mcp_policies.json:/app/mcp_policies.json
```

Run the service:

```bash
docker-compose up -d
```

#### Configure `mcp.json` for AI Integration

```json
{
  "mcpServers": {
    "vector_mcp": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "vector-mcp",
        "vector-mcp"
      ],
      "env": {
        "DATABASE_TYPE": "chromadb",                   // Optional
        "COLLECTION_NAME": "memory",                   // Optional
        "DOCUMENT_DIRECTORY": "/home/user/Documents/"  // Optional
      },
      "timeout": 300000
    }
  }
}

```

</details>

<details>
  <summary><b>Installation Instructions:</b></summary>

Install Python Package

```bash
python -m pip install vector-mcp
```

PGVector dependencies

```bash
python -m pip install vector-mcp[pgvector]
```

All

```bash
python -m pip install vector-mcp[all]
```

or

```bash
uv pip install --upgrade vector-mcp[all]
```



</details>

<details>
  <summary><b>Repository Owners:</b></summary>


<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)
</details>

Special shoutouts to Microsoft Autogen V1 ♥️
