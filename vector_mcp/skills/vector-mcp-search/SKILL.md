---
name: vector-mcp-search
description: Vector Mcp Search capabilities for A2A Agent.
---
### Overview
This skill provides access to search operations.

### Capabilities
- **search**: Performs a hybrid search combining semantic (vector) and lexical (BM25) methods.
- **semantic_search**: Searches and gathers related knowledge from the vector database instance using the question variable.
- **lexical_search**: Performs a lexical search using BM25 algorithm.

### Common Tools
- `search`: Performs a hybrid search combining semantic (vector) and lexical (BM25) methods.
- `semantic_search`: Searches and gathers related knowledge from the vector database instance using the question variable.
- `lexical_search`: Performs a lexical search using BM25 algorithm.

### Usage Rules
- Use these tools when the user requests actions related to **search** or **retrieve** and document.
- Always interpret the output of these tools to provide a concise summary to the user.

### Example Prompts
- "Please search for the following information"
- "Please retrieve related documents to the following question"
