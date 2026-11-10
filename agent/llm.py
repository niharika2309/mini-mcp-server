"""
agent/llm.py
============

A tiny LLM client layer. No heavyweight SDKs - just `requests` calling the
providers' REST APIs, because the interesting part of an agent is the message
and tool-call *format*, and it is easier to learn by reading ~250 lines than
by digging into an SDK.

One normalised conversation format is used internally (OpenAI-flavoured):

    {"role": "system"|"user", "content": str}
    {"role": "assistant", "content": str | None,
     "tool_calls": [{"id": str, "name": str, "arguments": dict}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Each provider translates that format to/from its own wire format and returns
a normalised `LLMResponse` (text and/or tool calls). `create_provider()`
builds the right one from the environment.

Providers:
  * OpenAICompatibleProvider - /chat/completions. Works with OpenAI and every
    OpenAI-compatible server (OpenRouter, Ollama, LM Studio, vLLM, ...) by
    pointing LLM_BASE_URL at it.
  * AnthropicProvider       - /v1/messages with tool_use content blocks.
  * MockProvider            - replays a scripted list of replies. No network,
    no key: used by the tests and the `--demo` CLI mode.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import AgentConfig, ConfigError

logger = logging.getLogger("agent.llm")


class LLMError(Exception):
    """A readable LLM problem (bad key, bad base URL, rate limit, malformed reply)."""


@dataclass
class ToolSpec:
    """One discovered MCP tool, in the shape the agent hands to the provider."""

    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    parse_error: Optional[str] = None  # set when the model emitted invalid JSON args


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)


Message = Dict[str, Any]  # normalised conversation message (see module docstring)


def _safe_json_object(raw: Any) -> tuple:
    """Parse tool-call arguments into a dict; return (dict, error_message_or_None)."""
    if isinstance(raw, dict):
        return raw, None
    if raw in (None, ""):
        return {}, None
    if not isinstance(raw, str):
        return {}, f"arguments must be a JSON object, got {type(raw).__name__}"
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        return {}, f"arguments are not valid JSON ({exc})"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object, not " + type(parsed).__name__
    return parsed, None


# ---------------------------------------------------------------------------
# Shared HTTP plumbing (retries + friendly error mapping)
# ---------------------------------------------------------------------------
class _HttpProvider:
    """Base class: performs POSTs with retries and maps failures to LLMError."""

    def __init__(self, config: AgentConfig, api_key_header: Dict[str, str]):
        self.config = config
        self._headers = {
            "Content-Type": "application/json",
            "User-Agent": "mini-mcp-agent/1.0",
            **api_key_header,
        }
        self._session = requests.Session()  # tests replace this with a fake

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[str] = None
        for attempt in range(self.config.max_retries + 1):
            if attempt:
                delay = 1.5 * attempt
                logger.warning("LLM retry %d in %.1fs (%s)", attempt, delay, last_error)
                time.sleep(delay)
            try:
                response = self._session.post(
                    url,
                    headers=self._headers,
                    json=payload,
                    timeout=self.config.request_timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    raise LLMError(f"The LLM endpoint returned a non-JSON response from {url}.")
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                continue

            raise LLMError(self._explain_http_error(response, url))

        raise LLMError(
            f"The LLM endpoint kept failing after {self.config.max_retries + 1} attempts "
            f"({last_error}). Check LLM_BASE_URL, your network, or try again later."
        )

    def _explain_http_error(self, response: requests.Response, url: str) -> str:
        status = response.status_code
        try:
            detail = response.json()
            detail = detail.get("error", detail) if isinstance(detail, dict) else detail
            message = (detail.get("message") if isinstance(detail, dict) else str(detail)) or ""
        except ValueError:
            message = response.text[:200]
        if status in (401, 403):
            return (
                f"The LLM provider rejected the API key (HTTP {status}). "
                "Check LLM_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY) in your .env."
            )
        if status == 404:
            return (
                f"HTTP 404 from {url}. Either LLM_MODEL ({self.config.model}) does not "
                "exist or LLM_BASE_URL points at the wrong server."
            )
        if status == 429:
            return f"Rate limited by the LLM provider (429): {message}"
        return f"LLM request failed (HTTP {status}): {message}"


# ---------------------------------------------------------------------------
# OpenAI-compatible /chat/completions
# ---------------------------------------------------------------------------
class OpenAICompatibleProvider(_HttpProvider):
    name = "openai"

    def __init__(self, config: AgentConfig):
        super().__init__(config, {"Authorization": f"Bearer {config.api_key}"})

    def _endpoint(self) -> str:
        base = self.config.base_url
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _to_wire(self, messages: List[Message], tools: List[ToolSpec]) -> Dict[str, Any]:
        wire_messages: List[Dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role in ("system", "user"):
                wire_messages.append({"role": role, "content": message["content"]})
            elif role == "assistant":
                entry: Dict[str, Any] = {"role": "assistant", "content": message.get("content")}
                if message.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                            },
                        }
                        for tc in message["tool_calls"]
                    ]
                wire_messages.append(entry)
            elif role == "tool":
                wire_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": message["content"],
                    }
                )
            else:  # defensive: unknown roles would break the API
                raise LLMError(f"Unsupported message role {role!r} (agent bug).")

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": wire_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in tools
            ]
        return payload

    def generate(self, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        data = self._post_json(self._endpoint(), self._to_wire(messages, tools))
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response shape from the LLM: {str(data)[:200]}")

        tool_calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments, parse_error = _safe_json_object(function.get("arguments"))
            tool_calls.append(
                ToolCall(
                    id=call.get("id") or f"call_{index}",
                    name=function.get("name") or "",
                    arguments=arguments,
                    parse_error=parse_error,
                )
            )
        return LLMResponse(text=(message.get("content") or "").strip(), tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Anthropic /v1/messages
# ---------------------------------------------------------------------------
class AnthropicProvider(_HttpProvider):
    name = "anthropic"

    def __init__(self, config: AgentConfig):
        super().__init__(
            config, {"x-api-key": config.api_key, "anthropic-version": "2023-06-01"}
        )

    def _endpoint(self) -> str:
        base = self.config.base_url
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    def _to_wire(self, messages: List[Message], tools: List[ToolSpec]) -> Dict[str, Any]:
        system_parts: List[str] = []
        wire_messages: List[Dict[str, Any]] = []

        for message in messages:
            role = message["role"]
            if role == "system":
                system_parts.append(message["content"])
            elif role == "user":
                wire_messages.append({"role": "user", "content": message["content"]})
            elif role == "assistant":
                blocks: List[Dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for tc in message.get("tool_calls") or []:
                    blocks.append(
                        {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                    )
                wire_messages.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                # Tool results travel as a user message holding tool_result blocks.
                wire_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message["tool_call_id"],
                                "content": message["content"],
                            }
                        ],
                    }
                )
            else:
                raise LLMError(f"Unsupported message role {role!r} (agent bug).")

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": wire_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        return payload

    def generate(self, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        data = self._post_json(self._endpoint(), self._to_wire(messages, tools))
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise LLMError(f"Unexpected response shape from the LLM: {str(data)[:200]}")

        texts, tool_calls = [], []
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                texts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                arguments, parse_error = _safe_json_object(block.get("input"))
                tool_calls.append(
                    ToolCall(
                        id=block.get("id") or f"toolu_{index}",
                        name=block.get("name") or "",
                        arguments=arguments,
                        parse_error=parse_error,
                    )
                )
        return LLMResponse(text="\n".join(texts).strip(), tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Mock provider - scripted replies, no network. Used by tests and `--demo`.
# ---------------------------------------------------------------------------
class MockProvider:
    """Replays a fixed script of assistant turns so the agent loop can run and
    be tested without any API key. Each scripted turn is a dict:

        {"text": "...", "tool_calls": [{"name": "...", "arguments": {...}}]}

    When the script runs out the provider gives a final wrap-up message, so
    the agent always terminates.
    """

    name = "mock"

    def __init__(self, script: Optional[List[Dict[str, Any]]] = None, script_path: str = ""):
        if script is None and script_path:
            with open(script_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, list):
                raise ConfigError("AGENT_MOCK_SCRIPT must contain a JSON list of turns.")
            script = loaded
        self.script = list(script or [])
        self._had_script = bool(self.script)
        self.calls: List[List[Message]] = []  # what the agent actually sent (tests assert on this)
        self.last_tools: List[ToolSpec] = []  # tool catalogue the agent passed in

    def generate(self, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        self.calls.append(messages)
        self.last_tools = list(tools)
        if not self.script:
            if self._had_script:
                return LLMResponse(text="(mock) Script finished - here is the summary of what the tools returned.")
            return LLMResponse(
                text="(mock provider) No script loaded - run `python -m agent --demo` "
                "to watch the scripted Week-11 walkthrough."
            )
        turn = self.script.pop(0)
        tool_calls = []
        for index, call in enumerate(turn.get("tool_calls") or []):
            arguments, parse_error = _safe_json_object(call.get("arguments"))
            tool_calls.append(
                ToolCall(
                    id=call.get("id") or f"mock_call_{index}",
                    name=call.get("name") or "",
                    arguments=arguments,
                    parse_error=parse_error,
                )
            )
        return LLMResponse(text=turn.get("text") or "", tool_calls=tool_calls)


def create_provider(config: AgentConfig):
    """Build the provider selected by LLM_PROVIDER (validated here so tools
    like --list-tools can run without a key until an LLM is actually needed)."""
    if config.provider == "openai":
        if not config.api_key:
            raise ConfigError(
                "No API key found for provider 'openai'. Set LLM_API_KEY or "
                "OPENAI_API_KEY in your environment or .env file."
            )
        return OpenAICompatibleProvider(config)
    if config.provider == "anthropic":
        if not config.api_key:
            raise ConfigError(
                "No API key found for provider 'anthropic'. Set LLM_API_KEY or "
                "ANTHROPIC_API_KEY in your environment or .env file."
            )
        return AnthropicProvider(config)
    if config.provider == "mock":
        return MockProvider(script_path=config.mock_script or "")
    raise ConfigError(f"Unknown provider {config.provider!r}.")
