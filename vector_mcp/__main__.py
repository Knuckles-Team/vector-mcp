"""``python -m vector_mcp`` runs the MCP server.

The A2A/LLM-agent entrypoint (``agent_server.py``) was removed in the
agent-connector-sdk migration (EH-484) — agent_connector_sdk has no equivalent to
agent-utilities' agent-runtime (``create_agent_server`` et al.); see
/var/tmp/l9/finish/au-decon-G4e/SDK-GAPS.md.
"""

from __future__ import annotations

from vector_mcp.mcp_server import mcp_server

if __name__ == "__main__":
    mcp_server()
