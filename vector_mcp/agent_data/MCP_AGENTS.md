# MCP_AGENTS.md - Dynamic Agent Registry

This file tracks the generated agents from MCP servers. You can manually modify the 'Tools' list to customize agent expertise.

## Agent Mapping Table

| Name | Description | System Prompt | Tools | Tag | Source MCP |
|------|-------------|---------------|-------|-----|------------|
| Vector Search Specialist | Expert specialist for search domain tasks. | You are a Vector Search specialist. Help users manage and interact with Search functionality using the available tools. | vector-mcp_search_toolset | search | vector-mcp |
| Vector Misc Specialist | Expert specialist for misc domain tasks. | You are a Vector Misc specialist. Help users manage and interact with Misc functionality using the available tools. | vector-mcp_misc_toolset | misc | vector-mcp |
| Vector Collection Management Specialist | Expert specialist for collection_management domain tasks. | You are a Vector Collection Management specialist. Help users manage and interact with Collection Management functionality using the available tools. | vector-mcp_collection_management_toolset | collection_management | vector-mcp |

## Tool Inventory Table

| Tool Name | Description | Tag | Source |
|-----------|-------------|-----|--------|
| vector-mcp_search_toolset | Static hint toolset for search based on config env. | search | vector-mcp |
| vector-mcp_misc_toolset | Static hint toolset for misc based on config env. | misc | vector-mcp |
| vector-mcp_collection_management_toolset | Static hint toolset for collection_management based on config env. | collection_management | vector-mcp |
