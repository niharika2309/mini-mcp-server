"""
agent/config.py
===============

Every knob of the agent is configured through environment variables (or a
`.env` file next to server.py, shared with the server). Nothing secret is
hardcoded: the LLM API key is read from the environment and only ever placed
into an HTTPS request header.

LLM selection
-------------
    LLM_PROVIDER      openai | anthropic | mock          (default: openai)
    LLM_MODEL         model name                          (default: per provider;
                      openai also honours OPENAI_MODEL / MODEL as fallbacks)
    LLM_API_KEY       the key; falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY
    LLM_BASE_URL      for OpenAI-compatible endpoints (OpenRouter, Ollama, ...)
                      (openai also honours OPENAI_BASE_URL as a fallback)
    LLM_TEMPERATURE   sampling temperature                (default: 0.2)
    LLM_TIMEOUT_SECONDS  per-request timeout              (default: 60)
    LLM_MAX_RETRIES   retries on timeouts / 429 / 5xx     (default: 2)
    LLM_MAX_TOKENS    response budget for the LLM         (default: 8192)

Agent behaviour
---------------
    AGENT_MAX_STEPS            tool-loop limit per user turn  (default: 10)
    AGENT_MAX_TOOL_RESULT_CHARS  truncation guard for results (default: 16000)
    AGENT_APPROVAL_TOOLS       comma-separated tool names that always require
                               a human approval prompt         (default: publish_article_to_github)
    AGENT_SERVER_COMMAND       override how the MCP server is launched
                               (default: this interpreter + server.py)
    AGENT_MOCK_SCRIPT          JSON file with canned replies for the mock provider
    AGENT_APPROVAL_TIMEOUT     web UI: seconds to wait for an Approve/Deny
                               click before a dangerous call is auto-denied
                               (default: 300)
    AGENT_LOG_LEVEL            DEBUG | INFO | WARNING (default: WARNING)
    AGENT_LOG_FILE             log file path (default: agent.log)

The mock provider needs no key and no network - it replays a script and is
what the test-suite and the offline demo (`python -m agent --demo`) use.
"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same .env the server reads, so one file configures both sides.
load_dotenv(REPO_ROOT / ".env")

VALID_PROVIDERS = ("openai", "anthropic", "mock")

# Sensible defaults per provider; LLM_MODEL always wins.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
    "mock": "mock-agent",
}
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "mock": "",
}
# Where the key may hide if LLM_API_KEY is not set, per provider.
KEY_FALLBACKS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "mock": (),
}


class ConfigError(Exception):
    """A configuration problem we can explain and fix (missing key, bad provider...)."""


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not an integer.")


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a number.")


@dataclass
class AgentConfig:
    """All agent settings in one validated object (built via AgentConfig.from_env())."""

    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float
    request_timeout: int
    max_retries: int
    max_tokens: int
    max_steps: int
    max_tool_result_chars: int
    approval_tools: Set[str]
    server_command: List[str]
    log_level: str
    log_file: Path
    mock_script: str = field(default="")
    approval_timeout: int = 300

    @classmethod
    def from_env(cls, require_api_key: bool = True) -> "AgentConfig":
        provider = _env("LLM_PROVIDER", "openai").lower()
        if provider not in VALID_PROVIDERS:
            raise ConfigError(
                f"LLM_PROVIDER={provider!r} is not supported. "
                f"Use one of: {', '.join(VALID_PROVIDERS)}."
            )

        model = _env("LLM_MODEL")
        if not model and provider == "openai":
            # Common alternative names people already have set for OpenAI-compatible servers.
            model = _env("OPENAI_MODEL") or _env("MODEL")
        if not model:
            model = DEFAULT_MODELS[provider]
        base_url = _env("LLM_BASE_URL")
        if not base_url and provider == "openai":
            base_url = _env("OPENAI_BASE_URL")
        base_url = base_url or DEFAULT_BASE_URLS[provider]

        api_key = _env("LLM_API_KEY")
        if not api_key:
            for fallback in KEY_FALLBACKS[provider]:
                api_key = _env(fallback)
                if api_key:
                    break
        if require_api_key and provider != "mock" and not api_key:
            names = ", ".join((("LLM_API_KEY",) + KEY_FALLBACKS[provider]))
            raise ConfigError(
                f"No API key found for provider '{provider}'. Set one of "
                f"{names} in your environment or in the .env file "
                "(LLM_PROVIDER=mock runs the agent offline with no key)."
            )

        # How the MCP server child process is launched. The default uses the
        # SAME interpreter that runs the agent, so one venv serves both.
        override = _env("AGENT_SERVER_COMMAND")
        if override:
            server_command = shlex.split(override)
        else:
            server_command = [sys.executable, str(REPO_ROOT / "server.py")]

        approval_tools = {
            name.strip()
            for name in _env("AGENT_APPROVAL_TOOLS", "publish_article_to_github").split(",")
            if name.strip()
        }

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            request_timeout=_env_int("LLM_TIMEOUT_SECONDS", 60),
            max_retries=_env_int("LLM_MAX_RETRIES", 2),
            max_tokens=_env_int("LLM_MAX_TOKENS", 8192),
            max_steps=_env_int("AGENT_MAX_STEPS", 10),
            max_tool_result_chars=_env_int("AGENT_MAX_TOOL_RESULT_CHARS", 16000),
            approval_tools=approval_tools,
            server_command=server_command,
            log_level=_env("AGENT_LOG_LEVEL", "WARNING").upper(),
            log_file=Path(_env("AGENT_LOG_FILE", "agent.log")),
            mock_script=_env("AGENT_MOCK_SCRIPT"),
            approval_timeout=_env_int("AGENT_APPROVAL_TIMEOUT", 300),
        )
