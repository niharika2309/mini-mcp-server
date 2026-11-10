"""
agent/cli.py
============

Interactive CLI for the agent: a conversation loop with visible progress.

    python -m agent                       # interactive REPL
    python -m agent "Find my latest Week 11 article and preview it"
    python -m agent --list-tools          # just prove tool discovery works
    python -m agent --demo                # offline scripted demo (no API key)
    python -m agent --web                 # browser UI (see agent/web.py)
    python -m agent --web --demo          # browser UI, scripted, no API key

What you see per turn:

    you ▸ find my latest week 11 article and check if it's in GitHub
    ⚙ calling list_substack_articles {"search": "Week 11", "limit": 5}
    ✔ list_substack_articles -> ok (1 article)
    ⚙ calling check_existing_article {"article": "Week 11 Day 3"}
    ✔ check_existing_article -> ok
    agent ▸ I found "Week 11, Day 3 - MCP ...". It is NOT in GitHub yet...

    ⚠ publish_article_to_github wants to write to GitHub:
        {"article": "Week 11 Day 3", "confirm": true}
      This creates exactly one .md file at the path shown in the preview.
    Approve this tool call? [y/N]:

Nothing dangerous runs without that y/N - and the server double-checks too.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Optional

from .agent import Agent
from .config import AgentConfig, ConfigError
from .llm import LLMError, create_provider
from .mcp_client import McpClient

# ---------------------------------------------------------------------------
# Tiny console helpers (plain unicode, ANSI only when attached to a terminal)
# ---------------------------------------------------------------------------
_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _short(value, limit: int = 110) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def render_event(event: str, data: Dict) -> None:
    """Print one agent-loop event. This is the 'show your work' layer."""
    if event == "assistant_text":
        print(_c("2", _short(data["text"], 400)))
    elif event == "tool_call":
        print(_c("36", f"  ⚙ calling {data['name']}"), _short(data["arguments"]))
    elif event == "tool_result":
        symbol = "✔" if data["ok"] else "✖"
        colour = "32" if data["ok"] else "31"
        print(_c(colour, f"  {symbol} {data['name']}"), "->", _short(data["payload"], 140))
    elif event == "tool_error":
        print(_c("33", f"  ✖ {data['name']} rejected"), "->", _short(data["error"], 140))
    elif event == "approval_denied":
        print(_c("31", f"  ⛔ {data['name']} DENIED by user - the call was not sent to the server"))
    elif event == "approval_confirmed":
        print(_c("2", "    (human approved - setting confirm=true for the server's own check)"))
    elif event == "limit":
        print(_c("33", f"  ⏹ {data['notice']}"))


class ConsoleAgent(Agent):
    """Agent with a y/N approval prompt wired to stdin."""

    async def _ask_human(self, call) -> bool:
        tool = self.mcp.tool(call.name)
        writes = "confirm" in (tool.input_schema.get("properties", {}) if tool else {})
        print()
        print(_c("1;33", f"  ⚠ {call.name} wants to run and needs your approval:"))
        print(_c("33", f"    arguments: {_short(call.arguments, 600)}"))
        if writes:
            print(
                _c(
                    "33",
                    "    This can WRITE to GitHub (a preview step should have shown the exact"
                    " file path and content first).",
                )
            )
        prompt = _c("1;33", "    Approve this tool call? [y/N]: ")
        try:
            answer = (await asyncio.to_thread(input, prompt)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""  # fail closed: no answer means no approval
        return answer in {"y", "yes"}


HELP_TEXT = """\
Commands:
  /help          show this help
  /tools         list the tools discovered from the MCP server
  /history       show how many messages are in the conversation context
  /reset         start a fresh conversation (keeps the MCP connection)
  /quit          exit (also Ctrl-D)
Anything else is sent to the agent.
"""

DEMO_TASK = (
    "Find my latest Week 11 Substack article, retrieve it, check whether it "
    "exists in GitHub, prepare a Markdown preview, and publish it once I approve."
)


def _setup_logging(config: AgentConfig, verbose: bool) -> None:
    """Logs go to a file (always) and to stderr (only with --verbose), so the
    REPL stays readable while agent.log keeps a full trail. API keys are never
    part of any log line - only request/tool metadata is logged."""
    level = logging.DEBUG if verbose else getattr(logging, config.log_level, logging.WARNING)
    handlers = [logging.FileHandler(config.log_file, encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent",
        description="AI agent that drives the mini-mcp-server through MCP.",
    )
    parser.add_argument("prompt", nargs="*", help="single question to run and exit")
    parser.add_argument("--list-tools", action="store_true", help="print discovered tools and exit")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="offline scripted run (LLM_PROVIDER=mock, canned Week-11 workflow, no API key)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="serve the browser UI (chat + live tool timeline + Approve/Deny) instead of the REPL",
    )
    parser.add_argument("--port", type=int, default=8765, help="port for --web (default: 8765)")
    parser.add_argument(
        "--no-browser", action="store_true", help="with --web: do not open the browser automatically"
    )
    parser.add_argument("--provider", help="override LLM_PROVIDER (openai|anthropic|mock)")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("--max-steps", type=int, help="override AGENT_MAX_STEPS")
    parser.add_argument("--verbose", "-v", action="store_true", help="log to stderr as well")
    return parser


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.demo:  # fully offline walkthrough, no key needed
        os.environ["LLM_PROVIDER"] = "mock"
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.max_steps:
        os.environ["AGENT_MAX_STEPS"] = str(args.max_steps)


async def _run(args: argparse.Namespace) -> int:
    _apply_cli_overrides(args)

    config = AgentConfig.from_env(require_api_key=not args.list_tools)
    _setup_logging(config, args.verbose)

    print(
        _c(
            "1",
            "mini-mcp agent",
        ),
        f"provider={config.provider} model={config.model}",
        f"| server: {' '.join(config.server_command)}",
    )

    # 1) connect + discover (works with no LLM key at all)
    mcp = McpClient(config.server_command)
    try:
        await mcp.start()
    except Exception as exc:
        print(_c("31", f"Could not start the MCP server: {exc}"), file=sys.stderr)
        print("Check that the venv is installed (pip install -r requirements.txt).", file=sys.stderr)
        return 1

    print(_c("36", f"MCP server {mcp.server_info!r} online - {len(mcp.tools)} tools discovered:"))
    for tool in mcp.tools:
        first_line = tool.description.split("\n")[0] if tool.description else ""
        required = tool.input_schema.get("required", [])
        print(f"  • {tool.name}({', '.join(tool.input_schema.get('properties', {}))})" + (f"  [required: {', '.join(required)}]" if required else ""))
        if first_line:
            print(_c("2", f"    {first_line}"))

    if args.list_tools:
        await mcp.close()
        return 0

    # 2) LLM provider (key is only needed from here on)
    try:
        if args.demo:
            provider = demo_provider()
        else:
            provider = create_provider(config)
    except (ConfigError, FileNotFoundError) as exc:
        print(_c("31", str(exc)), file=sys.stderr)
        await mcp.close()
        return 2

    agent = ConsoleAgent(provider, mcp, config, on_event=render_event)

    one_shot = " ".join(args.prompt).strip()
    if not one_shot and args.demo:
        one_shot = DEMO_TASK  # --demo runs the example task immediately
    if one_shot:
        try:
            answer = await agent.run_turn(one_shot)
            print(_c("1", "\nagent ▸"), answer)
        except LLMError as exc:
            print(_c("31", f"\nLLM error: {exc}"), file=sys.stderr)
            await mcp.close()
            return 3
        await mcp.close()
        return 0

    # 3) interactive conversation loop
    print(_c("2", "Type /help for commands, /quit to exit.\n"))
    while True:
        try:
            user_text = (await asyncio.to_thread(input, _c("1;36", "you ▸ "))).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in {"/quit", "/exit"}:
            break
        if user_text == "/help":
            print(HELP_TEXT)
            continue
        if user_text == "/tools":
            for tool in mcp.tools:
                print(f"  • {tool.name}")
            continue
        if user_text == "/history":
            print(f"conversation context: {len(agent.messages)} messages")
            continue
        if user_text == "/reset":
            agent.messages = agent.messages[:1]
            print("(conversation reset)")
            continue

        try:
            answer = await agent.run_turn(user_text)
        except LLMError as exc:
            print(_c("31", f"LLM error: {exc}"), file=sys.stderr)
            continue  # the REPL survives LLM failures
        except KeyboardInterrupt:
            print(_c("33", "(interrupted)"))
            continue
        print(_c("1", "\nagent ▸"), answer, "\n")

    await mcp.close()
    return 0


def demo_provider():
    """The Week-11 example task as a script, so `python -m agent --demo` shows
    the whole flow (discover -> read -> check -> preview -> approve -> publish)
    with zero credentials. Approval still prompts (terminal y/N in the CLI,
    buttons in --web); the publish call hits the server's real confirm and
    duplicate guards."""
    from .llm import MockProvider

    return MockProvider(
        script=[
            {
                "text": "First, let me find your Week 11 articles on Substack.",
                "tool_calls": [{"name": "list_substack_articles", "arguments": {"search": "Week 11", "limit": 5}}],
            },
            {
                "tool_calls": [{"name": "get_substack_article", "arguments": {"article": "Week 11 Day 3"}}],
            },
            {
                "tool_calls": [{"name": "check_existing_article", "arguments": {"article": "Week 11 Day 3"}}],
            },
            {
                "tool_calls": [{"name": "preview_article", "arguments": {"article": "Week 11 Day 3"}}],
            },
            {
                "text": "The preview looks good and nothing exists at that path yet. Publishing now (needs your approval).",
                "tool_calls": [
                    {"name": "publish_article_to_github", "arguments": {"article": "Week 11 Day 3", "confirm": True}}
                ],
            },
            {"text": "Done - see the tool results above for the file URL (or the refusal, if approval or GITHUB_TOKEN was missing)."},
        ]
    )


def _web_mode(args: argparse.Namespace) -> int:
    """Browser UI mode: same agent, served at http://127.0.0.1:<port>."""
    _apply_cli_overrides(args)
    config = AgentConfig.from_env(require_api_key=not args.demo)
    _setup_logging(config, args.verbose)

    try:
        provider = demo_provider() if args.demo else create_provider(config)
    except (ConfigError, FileNotFoundError) as exc:
        print(_c("31", str(exc)), file=sys.stderr)
        return 2

    print(
        _c("1", "mini-mcp agent"),
        f"provider={config.provider} model={config.model}",
        f"| server: {' '.join(config.server_command)}",
    )
    from . import web

    mcp = McpClient(config.server_command)
    return web.serve(args, config, mcp, provider, demo=args.demo, open_browser=not args.no_browser)


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.web:
        try:
            return _web_mode(args)
        except KeyboardInterrupt:
            return 130
    try:
        return asyncio.run(_run(args))
    except ConfigError as exc:
        print(_c("31", f"Configuration error: {exc}"), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
