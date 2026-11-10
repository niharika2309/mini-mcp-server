"""
agent
=====

An AI agent that talks to the mini-mcp-server over MCP (stdio) and drives it
with a configurable LLM.

Layering (the server and the agent are deliberately separate):

    user (CLI)  ->  agent/  ->  [MCP over stdio]  ->  server.py  ->  Substack / GitHub

The agent NEVER imports server.py directly; it discovers the tools at runtime
with a `tools/list` request, exactly like Claude Desktop would. That keeps the
server independently usable and the server/agent boundary honest.
"""

__all__ = ["config", "llm", "mcp_client", "agent"]
