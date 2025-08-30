# Vector Database MCP Server

![PyPI - Version](https://img.shields.io/pypi/v/repository-manager)
![PyPI - Downloads](https://img.shields.io/pypi/dd/repository-manager)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/repository-manager)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/repository-manager)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/repository-manager)
![PyPI - License](https://img.shields.io/pypi/l/repository-manager)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/repository-manager)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/repository-manager)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/repository-manager)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/repository-manager)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/repository-manager)

![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/repository-manager)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/repository-manager)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/repository-manager)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/repository-manager)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/repository-manager)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/repository-manager)

*Version: 0.0.2*

Connect to PGVector database and create collections of memory storage for AI Agents.

AI Agents can:

- Create collections with documents
- Update existing documents in a collection
- Delete collection
- Utilize collection for retrieval augmented generation (RAG)
- Leverage Euclidean or Cosine search

#### Using as an MCP Server:

AI Prompt:
```text
Clone all the git projects located in the file "/home/genius/Development/repositories-list/repositories.txt" to my "/home/genius/Development" folder.
Afterwards, pull all the projects located in the "/home/genius/Development" repository folder.
```

AI Response:
```text
All projects in "/home/genius/Development/repositories-list/repositories.txt" have been cloned to "/home/genius/Development"
and all projects in "/home/genius/Development" and been pulled from the repositories. Let me know if you need any further actions! 🚀.
```

This repository is actively maintained - Contributions are welcome!

<details>
  <summary><b>Usage:</b></summary>

| Short Flag | Long Flag        | Description                            |
|------------|------------------|----------------------------------------|
| -h         | --help           | See Usage                              |
| -b         | --default-branch | Checkout default branch                |
| -c         | --clone          | Clone projects specified               |
| -d         | --directory      | Directory to clone/pull projects       |
| -f         | --file           | File with repository links             |
| -p         | --pull           | Pull projects in parent directory      |
| -r         | --repositories   | Comma separated Git URLs               |
| -t         | --threads        | Number of parallel threads - Default 4 |

</details>

<details>
  <summary><b>Example:</b></summary>

### Use in CLI

```bash
repository-manager \
    --clone  \
    --pull  \
    --directory '/home/user/Downloads'  \
    --file '/home/user/Downloads/repositories.txt'  \
    --repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot' \
    --threads 8
```

### Use in Python

```python
from repository_manager import Git

gitlab = Git()

gitlab.set_repository_directory("<directory>")

gitlab.set_threads(threads=8)

gitlab.set_git_projects("<projects>")

gitlab.set_default_branch(set_to_default_branch=True)

gitlab.clone_projects_in_parallel()

gitlab.pull_projects_in_parallel()
```

### Use with AI

Deploy MCP Server as a Service
```bash
docker pull knucklessg1/repository-manager:latest
```

Modify the `compose.yml`

```compose
services:
  repository-manager-mcp:
    image: knucklessg1/repository-manager:latest
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
    "repository_manager": {
      "command": "repository-manager-mcp",
      "env": {
        "REPOSITORY_DIRECTORY": "/home/user/Development/",            // Optional - Can be specified at prompt
        "THREADS": "12",                                              // Optional - Can be specified at prompt
        "DEFAULT_BRANCH": "True",                                     // Optional - Can be specified at prompt
        "PROJECTS_FILE": "/home/user/Development/repositories.txt"    // Optional - Can be specified at prompt
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
python -m pip install repository-manager
```
</details>

## Geniusbot Application

Use with a GUI through Geniusbot

Visit our [GitHub](https://github.com/Knuckles-Team/geniusbot) for more information

<details>
  <summary><b>Installation Instructions with Geniusbot:</b></summary>

Install Python Package

```bash
python -m pip install geniusbot
```

</details>


<details>
  <summary><b>Repository Owners:</b></summary>


<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)
</details>

Notes:

Additional Document and Function handling is based out of Microsoft's Autogen (v1) framework.
