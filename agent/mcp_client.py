"""
agent/mcp_client.py
===================

The thin wrapper around the official MCP SDK's client side. It does three
jobs and nothing else:

1. **Start** - spawn `python server.py` as a child process and speak MCP over
   its stdin/stdout (stdio transport), using `mcp.client.stdio.stdio_client`
   and `ClientSession` from the official SDK.
2. **Discover** - `list_tools()` once at startup and keep the tool catalogue
   (name, description, input schema) exactly as the server advertised it.
   The agent never hardcodes a tool definition - if the server adds a tool,
   the agent sees it automatically.
3. **Call** - `call_tool(name, arguments)`, normalising the SDK result into a
   simple `ToolResult(ok, payload, text)` the agent loop can reason about.

For tests, `session=` can be injected (an in-memory ClientSession connected to
the real server object), so the whole stack can run without child processes.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp import types as mcp_types

logger = logging.getLogger("agent.mcp")


@dataclass
class ToolInfo:
    """One tool exactly as discovered from the server's tools/list response."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    annotations: Optional[Dict[str, Any]] = None

    def is_destructive(self) -> bool:
        """Honour the MCP `destructiveHint` annotation when a server sets it."""
        return bool(self.annotations and self.annotations.get("destructiveHint"))


@dataclass
class ToolResult:
    """Normalised result of one tools/call round trip."""

    ok: bool                     # False = protocol-level tool error or crash
    payload: Any                 # parsed JSON dict (usually), else raw string
    text: str                    # raw text returned by the server


class McpClient:
    """Owns the MCP session + the discovered tool catalogue."""

    def __init__(
        self,
        server_command: List[str],
        env: Optional[Dict[str, str]] = None,
        client_name: str = "mini-mcp-agent",
        session: Optional[ClientSession] = None,
    ):
        self.server_command = list(server_command)
        self.env = env
        self.client_name = client_name
        self._injected_session = session
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self.tools: List[ToolInfo] = []
        self.instructions: str = ""   # the server's own usage instructions
        self.server_info: str = ""

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        if self._injected_session is not None:
            self._session = self._injected_session
        else:
            # AsyncExitStack keeps the child process tidy: when the session or
            # the stdio pipes fail, their cleanup (kill the server, close the
            # anyio task group) still runs.
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()
            params = StdioServerParameters(
                command=self.server_command[0],
                args=self.server_command[1:],
                env=self.env,
            )
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

        init = await self._session.initialize()
        self.server_info = getattr(init.server_info, "name", "unknown")
        self.instructions = getattr(init, "instructions", "") or ""
        logger.info("connected to MCP server %r (protocol %s)", self.server_info, init.protocol_version)

        self.tools = await self._discover_tools()
        logger.info("discovered %d tools: %s", len(self.tools), ", ".join(t.name for t in self.tools))

    async def _discover_tools(self) -> List[ToolInfo]:
        assert self._session is not None
        tools: List[ToolInfo] = []
        cursor: Optional[str] = None
        while True:
            params = mcp_types.PaginatedRequestParams(cursor=cursor) if cursor else None
            page = await self._session.list_tools(params=params)
            for tool in page.tools:
                annotations = getattr(tool, "annotations", None)
                if annotations is not None and not isinstance(annotations, dict):
                    annotations = annotations.model_dump(exclude_none=True)
                tools.append(
                    ToolInfo(
                        name=tool.name,
                        description=(tool.description or "").strip(),
                        input_schema=dict(getattr(tool, "input_schema", None) or {}),
                        annotations=annotations,
                    )
                )
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                return tools

    async def close(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception as exc:  # closing a dead child should never crash the CLI
                logger.debug("error while closing MCP session: %s", exc)
            self._exit_stack = None
        self._session = None

    async def __aenter__(self) -> "McpClient":
        await self.start()
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.close()

    # -- calling ------------------------------------------------------------
    def tool(self, name: str) -> Optional[ToolInfo]:
        return next((t for t in self.tools if t.name == name), None)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """One tools/call, converted into ToolResult. Never raises: server-side
        errors and transport crashes both come back as ok=False so the agent
        loop can decide what to tell the model."""
        assert self._session is not None, "call start() first"
        logger.info("tools/call %s(%s)", name, _preview(arguments))
        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as exc:  # unknown tool, server died, timeout...
            return ToolResult(ok=False, payload={"error": f"MCP call failed: {exc}"}, text=str(exc))

        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict) and structured:
            payload: Any = structured
        elif text:
            try:
                payload = json.loads(text)
            except ValueError:
                payload = text  # plain-text result from a non-JSON tool
        else:
            payload = {"success": not result.is_error}

        return ToolResult(ok=not result.is_error, payload=payload, text=text or json.dumps(payload, default=str))


def _preview(value: Any, limit: int = 200) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "...(truncated)"
