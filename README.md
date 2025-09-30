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

*Version: 0.1.6*

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

## Using as an MCP Server:

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

</details>

<details>
  <summary><b>Example:</b></summary>

### Use in CLI


| Short Flag | Long Flag        | Description                   |
|------------|------------------|-------------------------------|
| -h         | --help           | See Usage                     |
| -h         | --host           | Host of Vector Database       |
| -p         | --port           | Port of Vector Database       |
| -d         | --path           | Path of local Vector Database |
| -t         | --transport      | Transport Type (https/stdio)  |

```bash
vector-mcp
```

### Use with AI

Deploy MCP Server as a Service
```bash
docker pull knucklessg1/vector-mcp:latest
```

Modify the `compose.yml`

```compose
services:
  vector-mcp-mcp:
    image: knucklessg1/vector-mcp:latest
    volumes:
      - development:/root/Development
    environment:
      - HOST=0.0.0.0
      - PORT=8001
    ports:
      - 8001:8001
```

Configure `mcp.json`

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


</details>

<details>
  <summary><b>Repository Owners:</b></summary>


<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)
</details>

Special shoutouts to Microsoft Autogen V1 ♥️
