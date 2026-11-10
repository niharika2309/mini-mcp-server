"""
agent/agent.py
==============

The agent execution loop - the heart of this layer.

For each user turn:

    1. append the user message to the conversation history
    2. send history + the DISCOVERED tool specs to the LLM
    3. if the LLM asks for tool calls:
         a. validate the generated arguments against the tool's JSON schema
            (a bad call never reaches the server; the error goes back to the
            model so it can self-correct)
         b. if the tool is dangerous (publish!): pause and ask a HUMAN
            (approval_callback). Denied -> the model is told, and must adapt.
         c. execute via MCP tools/call, append the (truncated) result,
            go back to 2 so the model can inspect it and decide the next step
    4. if the LLM answers with plain text (no tool calls) -> turn is done
    5. hard stop after AGENT_MAX_STEPS so a confused model cannot spin

Safeguard layering (defence in depth - nothing here weakens the server):
  * layer 1 (agent):   human approval gate before any dangerous tool runs
  * layer 2 (server):  publish_article_to_github still requires confirm=True,
    re-reads the file and requires overwrite=True, validates the path, etc.
When - and only when - a human has just approved a call, the agent sets
confirm=true if the model forgot it. It can NEVER make the server write
without the server's own checks passing too.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import jsonschema

from .config import AgentConfig
from .llm import LLMResponse, Message, ToolCall, ToolSpec

logger = logging.getLogger("agent.loop")

SYSTEM_PROMPT = """\
You are an AI assistant running as an agent on top of the "{server_name}" MCP server.
You can only act through the provided tools - never invent facts, file paths,
URLs or article content that a tool could fetch for you.

Rules of work:
- Break the request into steps: discover, read, verify, then (if asked) write.
- Call tools with arguments that exactly match each tool's input schema.
- After every tool result, inspect it before deciding the next action.
- If a result shows an error or a surprise, adapt - do not repeat the same call.
- If the request is ambiguous (which article? which path?), ask a short
  clarifying question as your final answer instead of guessing.
- You have NO access to GitHub or Substack except through these tools.
- Publishing changes the user's repository. The tool itself requires
  confirm=true AND the harness will ask the user to approve the call.
  Only set confirm=true when the user has clearly asked you to publish.
- Never claim something was published unless a tool result says so.
- When finished, reply to the user in clear plain language (no tool calls).
"""

# async callable(tool_name, arguments) -> bool  (True = human approved)
ApprovalCallback = Callable[[str, Dict[str, Any]], Awaitable[bool]]
# sync callable(event_type, data)  (CLI renders these as progress messages)
EventCallback = Callable[[str, Dict[str, Any]], None]


class Agent:
    """One agent = one MCP connection + one LLM provider + one conversation."""

    def __init__(
        self,
        provider,
        mcp_client,
        config: AgentConfig,
        approval_callback: Optional[ApprovalCallback] = None,
        on_event: Optional[EventCallback] = None,
    ):
        self.provider = provider
        self.mcp = mcp_client
        self.config = config
        self.approval_callback = approval_callback
        self.on_event = on_event or (lambda event, data: None)

        self.tools: List[ToolSpec] = [
            ToolSpec(name=t.name, description=t.description, input_schema=t.input_schema)
            for t in mcp_client.tools
        ]
        system = SYSTEM_PROMPT.format(server_name=mcp_client.server_info or "mini-mcp-server")
        if mcp_client.instructions:
            system += "\nThe MCP server describes itself as:\n" + mcp_client.instructions + "\n"
        # Conversation history: system prompt + alternating user/assistant/tool
        # messages; it persists across turns so "now publish it" can refer to
        # the article discussed earlier in the session.
        self.messages: List[Message] = [{"role": "system", "content": system}]

    # -- small helpers -------------------------------------------------------
    def _dangerous(self, tool_name: str) -> bool:
        tool = self.mcp.tool(tool_name)
        return tool_name in self.config.approval_tools or bool(tool and tool.is_destructive())

    def _emit(self, event: str, **data: Any) -> None:
        try:
            self.on_event(event, data)
        except Exception:  # a broken progress renderer must never kill the loop
            logger.exception("on_event callback failed")

    def _truncate(self, text: str) -> str:
        """Cap what the model sees: one article is fine, a whole repo dump is not.
        Truncation is explicit, so the model knows it is looking at a prefix."""
        limit = self.config.max_tool_result_chars
        if len(text) <= limit:
            return text
        logger.info("truncating tool result from %d to %d chars", len(text), limit)
        return json.dumps(
            {
                "truncated": True,
                "note": "The full result was too large to show; here is a prefix.",
                "total_chars": len(text),
                "preview": text[:limit],
            },
            ensure_ascii=False,
        )

    def _append_tool_message(self, call: ToolCall, content: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}
        )

    # -- the main loop ---------------------------------------------------------
    async def run_turn(self, user_text: str) -> str:
        """Process one user message until the model answers or limits trigger."""
        self.messages.append({"role": "user", "content": user_text})

        for step in range(1, self.config.max_steps + 1):
            logger.info("step %d/%d: calling %s", step, self.config.max_steps, self.provider.name)
            # Run the (blocking) HTTP call off the event loop so MCP traffic
            # can keep flowing while the model thinks.
            response: LLMResponse = await asyncio.to_thread(
                self.provider.generate, self.messages, self.tools
            )

            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                }
            )
            if response.text:
                self._emit("assistant_text", text=response.text)

            if not response.tool_calls:
                return response.text or "(The model returned an empty reply - try rephrasing.)"

            for call in response.tool_calls:
                await self._execute_tool_call(call)

        notice = (
            f"I stopped after {self.config.max_steps} tool steps without finishing. "
            "Try narrowing the request or raising AGENT_MAX_STEPS."
        )
        self._emit("limit", notice=notice)
        self.messages.append({"role": "assistant", "content": notice})
        return notice

    async def _execute_tool_call(self, call: ToolCall) -> None:
        tool = self.mcp.tool(call.name)
        if tool is None:
            # The model hallucinated a tool name: teach it what exists.
            known = ", ".join(t.name for t in self.tools) or "(none discovered)"
            self._append_tool_message(
                call,
                json.dumps(
                    {
                        "success": False,
                        "error": f"Unknown tool {call.name!r}. Tools available: {known}.",
                    }
                ),
            )
            self._emit("tool_error", name=call.name, error="unknown tool")
            return

        # 1) arguments must exist and be valid JSON (provider already parsed them)
        if call.parse_error:
            self._append_tool_message(
                call,
                json.dumps({"success": False, "error": f"Invalid tool arguments: {call.parse_error}."}),
            )
            self._emit("tool_error", name=call.name, error=call.parse_error)
            return

        # 2) schema validation against the DISCOVERED schema - the model gets
        #    the validation error back and can retry with a corrected call.
        #    Before giving up, retry once with scalar type repair ("2" -> 2):
        #    small models flip int/bool to strings constantly and resending
        #    the identical call wastes a step and can loop.
        validation_error = _validate_arguments(tool.input_schema, call.arguments)
        if validation_error:
            coerced = _coerce_arguments(tool.input_schema, call.arguments)
            if coerced != call.arguments and not _validate_arguments(tool.input_schema, coerced):
                logger.info("coerced scalar arguments for %s: %s", call.name, coerced)
                call.arguments = coerced
                validation_error = None
        if validation_error:
            logger.warning("schema rejection for %s: %s", call.name, validation_error)
            self._append_tool_message(
                call,
                json.dumps(
                    {
                        "success": False,
                        "error": "Your arguments did not match this tool's input schema.",
                        "schema_errors": validation_error,
                        "input_schema": tool.input_schema,
                    }
                ),
            )
            self._emit("tool_error", name=call.name, error=validation_error)
            return

        # 3) human approval gate for dangerous tools (publish!). Denied calls
        #    never reach the server. The approval_callback itself announces
        #    the request (the CLI prints a prompt, the web UI emits an event
        #    with an id for its Approve/Deny buttons).
        if self._dangerous(call.name):
            approved = await self._ask_human(call)
            if not approved:
                self._emit("approval_denied", name=call.name)
                self._append_tool_message(
                    call,
                    json.dumps(
                        {
                            "success": False,
                            "published": False,
                            "user_approved": False,
                            "error": (
                                "The user did NOT approve this action, so nothing was done. "
                                "Do not retry the same call; tell the user what you had "
                                "prepared and ask how they would like to proceed."
                            ),
                        }
                    ),
                )
                return
            # The human said yes -> make sure the server's own confirmation
            # flag is set if the model forgot it (layer 2 still runs fully).
            if "confirm" in tool.input_schema.get("properties", {}) and call.arguments.get("confirm") is not True:
                logger.info("human approved %s: setting confirm=true", call.name)
                call.arguments["confirm"] = True
                self._emit("approval_confirmed", name=call.name)

        # 4) execute over MCP
        self._emit("tool_call", name=call.name, arguments=call.arguments)
        result = await self.mcp.call_tool(call.name, call.arguments)

        raw = result.text if isinstance(result.payload, str) else json.dumps(
            result.payload, ensure_ascii=False, default=str
        )
        if result.ok:
            content = self._truncate(raw)
        else:
            # Flag protocol-level tool errors clearly so the model does not
            # mistake a failed call for useful data.
            content = self._truncate(
                json.dumps({"success": False, "tool_error": True, "result": raw}, ensure_ascii=False)
            )
        self._append_tool_message(call, content)
        self._emit("tool_result", name=call.name, ok=result.ok, payload=result.payload)

    async def _ask_human(self, call: ToolCall) -> bool:
        if self.approval_callback is None:
            # No way to ask -> fail closed. Publishing without a human is a
            # hard no, even if the model is confident.
            logger.warning("no approval channel available; denying %s", call.name)
            return False
        return bool(await self.approval_callback(call.name, dict(call.arguments)))


_TRUE_WORDS = {"true", "1", "yes", "y", "on"}
_FALSE_WORDS = {"false", "0", "no", "n", "off"}


def _coerce_value(schema: Dict[str, Any], value: Any) -> Any:
    """Repair the scalar type slips small tool-calling models make
    (limit:"2" instead of 2, confirm:"true" instead of True). The result is
    still validated against the real schema afterwards, so this can never
    turn an invalid call into a valid-but-wrong one."""
    if not isinstance(schema, dict):
        return value
    stype = schema.get("type")
    if stype == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        return {k: _coerce_value(props[k], v) if k in props else v for k, v in value.items()}
    if stype == "array" and isinstance(value, list):
        items = schema.get("items")
        return [_coerce_value(items, v) for v in value] if items else value
    if not isinstance(value, str) or not isinstance(stype, str):
        return value
    try:
        if stype == "integer" and value.strip().lstrip("+-").isdigit():
            return int(value)
        if stype == "number":
            return float(value)
        if stype == "boolean":
            word = value.strip().lower()
            if word in _TRUE_WORDS:
                return True
            if word in _FALSE_WORDS:
                return False
    except (ValueError, AttributeError):
        pass
    return value


def _coerce_arguments(schema: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce every argument whose name appears in the schema's properties."""
    if not schema:
        return arguments
    props = schema.get("properties", {})
    return {k: _coerce_value(props[k], v) if k in props else v for k, v in arguments.items()}


def _validate_arguments(schema: Dict[str, Any], arguments: Dict[str, Any]) -> Optional[str]:
    """Return None when arguments satisfy the tool's JSON schema, else a
    readable, newline-joined list of what the model must fix."""
    if not schema:
        return None
    validator_cls = jsonschema.validators.validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except jsonschema.SchemaError:
        return None  # a broken server schema is the server's problem, not the model's
    errors = sorted(
        validator_cls(schema).iter_errors(arguments), key=lambda e: list(e.absolute_path)
    )
    if not errors:
        return None
    lines = []
    for error in errors[:5]:
        where = ".".join(str(p) for p in error.absolute_path) or "(root)"
        lines.append(f"{where}: {error.message}")
    return "\n".join(lines)
