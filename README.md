# mini-mcp-server

Two things live in this repo, and they stay logically separate:

1. **An [MCP](#what-is-mcp-in-plain-english) server** that connects the two
   halves of Niharika's AI/ML education project:
   - **Substack** — the [16-week AI/ML series](https://niharikabatra.substack.com/s/16-week-aiml-series) where the articles are published
   - **GitHub** — the [16-week-ai-ml-series](https://github.com/niharika2309/16-week-ai-ml-series) curriculum repo where each article also lives as a Markdown file
2. **An AI agent** (`agent/`) that connects to that server *as an MCP client*
   and completes multi-step tasks with a configurable LLM — from the CLI:

   > *"Find my latest Week 11 Substack article, retrieve it, check whether it
   > exists in GitHub, prepare a preview, and publish only after I approve."*

With the **server** installed in any MCP client (Claude Desktop, Cursor, ChatGPT
Desktop, MCP Inspector, or the agent here), the assistant can do things like:

> *"List my published articles from the 16-week AI/ML series."*
> *"Fetch Week 11 Day 4."*
> *"Add this article to GitHub."* → it inspects the repo, checks for
> duplicates, shows a preview, and **asks you before writing anything.**

Everything read from Substack uses only public data (RSS feed + public post
pages). Writing to GitHub uses your `GITHUB_TOKEN`, and the server refuses to
commit without your explicit confirmation.

---

## What is MCP, in plain English?

Think of MCP (**Model Context Protocol**) as a **USB-C port for AI
assistants**. An assistant like Claude can't browse your GitHub repo or read
your Substack by itself — but if you plug an "MCP server" into it, the
assistant suddenly gets a set of named tools (`list_substack_articles`,
`publish_article_to_github`, ...) that it can discover and call.

The mechanics (and where each part lives in this project):

1. **You** add this server to your MCP client (Claude Desktop config, Cursor
   settings, ...). The client starts `python server.py` as a child process.
2. **Transport: stdio.** The client and server exchange small JSON-RPC
   messages through the child process's stdin/stdout. That's why the code
   never uses `print()` — stdout is the wire. Debug output goes to stderr via
   `_log()`.
3. **Discovery.** Each Python function decorated with `@mcp.tool()` is
   advertised to the assistant through a `tools/list` request. The assistant
   sees the tool's name, its description (from the docstring) and a JSON
   input schema generated from the type hints.
4. **Calling.** When you say *"fetch Week 11 Day 4"*, the assistant picks the
   matching tool and sends a `tools/call` request with arguments. The SDK
   runs the Python function, and the returned dictionary comes back to the
   assistant as the tool result.

One important safety idea runs through this project: **the only tool that
writes to GitHub (`publish_article_to_github`) is fenced behind an explicit
`confirm=True` flag**, and the tool's instructions tell the assistant it must
show you `preview_article` first and get your approval.

---

## Project architecture

There are two independent halves. The **server** is a normal MCP server that
any MCP client can use. The **agent** is one such client: it launches the
server over stdio, *discovers* the tools at runtime, and drives them with a
configurable LLM.

```
+---------------------------------------------------------------------------------------+
| agent layer (this repo: agent/)                                                        |
|   cli.py         conversation REPL, progress & tool-use messages, y/N approval prompt |
|   agent.py       loop: plan -> select tool -> generate args -> validate -> approve    |
|                  (dangerous tools) -> execute -> inspect result -> repeat or stop     |
|   llm.py         LLM providers: OpenAI-compatible / Anthropic / mock (env-configured) |
|   mcp_client.py  official MCP SDK client (stdio_client + ClientSession)               |
+---------------------------------------------------+-----------------------------------+
                                                    |  MCP over stdio
     AI assistant (Claude Desktop / Cursor / MCP      |  (JSON-RPC on stdin/stdout)
     Inspector) connects the SAME way as the agent    v
+-------------------------------------+
| server.py                           |   @mcp.tool() x6  <- the MCP layer:
|  (MCPServer, a.k.a. FastMCP)        |   names, schemas, docstrings,
|                                     |   confirmation + duplicate guards
+----------------+--------------------+
                 |                    |
                 v                    v
+----------------+----+  +-------------+--------+        +---------------------------+
| substack_reader.py  |  | github_writer.py     |------->| GitHub REST API           |
|  (requests, public  |  |  (requests + Bearer  |        | api.github.com            |
|   RSS + post data)  |  |   token from env)    |        | GITHUB_TOKEN (from .env)  |
+-------------------- +  +----------------------+        +---------------------------+
        |
        v
+----------------------+
| article_parser.py    |  HTML -> clean Markdown, repo file-naming rules
+----------------------+  (no magic library - the rules are readable code)
```

| File | Responsibility |
|---|---|
| `server.py` | Creates the MCP server, defines the six tools, enforces the confirmation workflow. |
| `substack_reader.py` | Reads the public RSS feed (article discovery) and public post data (article content). Never touches private/paid content. |
| `article_parser.py` | Converts article HTML to Markdown (headings, lists, links, code blocks, quotes, images) and computes repo destination paths. |
| `github_writer.py` | All GitHub REST API calls: list the repo tree, read one file, create/update one file. |
| `agent/config.py` | Every agent setting comes from environment variables (provider, model, key, limits...). |
| `agent/llm.py` | LLM providers over plain REST: OpenAI-compatible, Anthropic, plus an offline mock. One normalised tool-calling format. |
| `agent/mcp_client.py` | MCP client (official SDK): starts `server.py` over stdio, discovers tools, executes calls. |
| `agent/agent.py` | The agent loop + safeguards: schema validation, approval gate, step limit, result truncation. |
| `agent/cli.py` | The conversation REPL you actually run: `python -m agent`. |
| `agent/web.py` | The optional browser UI (`--web`): stdlib HTTP server + one embedded page; chat, SSE tool timeline, Approve/Deny buttons. |
| `tests/` | Pytest suite: loop behaviour, providers, config, web UI, and the full workflow against the real server code. |
| `requirements.txt` | Python dependencies (server *and* agent share one environment). |
| `requirements-dev.txt` | Test-only dependencies (pytest). |
| `.env.example` | Template for your secret `GITHUB_TOKEN` and the agent's LLM settings. |

## The six tools

| Tool | What it does | Writes to GitHub? |
|---|---|---|
| `list_substack_articles` | Lists published series articles (title, date, URL, description, week/day). Filters out non-series announcement posts; supports `search` + `limit`. | No |
| `get_substack_article` | Fetches one article (URL, slug, or "Week 11 Day 3") and returns it as clean Markdown. Flags subscriber-only previews. | No |
| `list_github_files` | Shows the repo structure (`01-week-1/`, `02-week-2/`, ...), optionally filtered to one folder. | No |
| `check_existing_article` | Answers *"does this article already exist in the repo?"* and lists sibling files in the week folder to catch near-duplicates. | No |
| `preview_article` | Shows the exact Markdown and destination path (`11-week-11/day-3-<slug>.md`) that would be created — the thing you should always read before approving. | No |
| `publish_article_to_github` | Commits the Markdown file via the GitHub contents API. **Refuses unless `confirm=True`** (and `overwrite=True` too if a file exists), so nothing is ever written without your explicit approval. | Yes — the only one |

Article files follow the repo's existing conventions exactly:

```
NN-week-NN/day-D-<slug>.md        e.g.  11-week-11/day-3-mcp-how-ai-agents-connect.md
```

with the same header style as the existing files:

```markdown
# Week 11, Day 3 - MCP : How AI Agents Connect to the Outside World

*November 6, 2026 · [Originally published on Substack](https://niharikabatra.substack.com/p/...)*
```

Subscriber-only posts are mirrored as their free public preview with a 🔒
banner linking back to Substack — exactly like the current repo files.

---

## Installation

You need **Python 3.10+** (3.11+ recommended).

```bash
cd mini-mcp-server

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# (optional, for the agent test-suite)
pip install -r requirements-dev.txt
```

## Configuration

```bash
cp .env.example .env
```

Then edit `.env` and set your token:

```bash
GITHUB_TOKEN=github_pat_...
```

**How to create the token:** open <https://github.com/settings/tokens> →
*Fine-grained tokens* → select repository `niharika2309/16-week-ai-ml-series`
→ permission **Contents: Read and write**. (A classic token with the `repo`
scope also works.) The token is read from the environment at runtime — it is
never hardcoded and never sent anywhere except `api.github.com`. `.env` is
git-ignored.

Optional overrides (defaults already match the curriculum repo):
`GITHUB_OWNER`, `GITHUB_REPO`, `GITHUB_DEFAULT_BRANCH`.

Running the **agent** additionally needs the `LLM_*` variables (provider,
model, API key, base URL) — the full table is in the **LLM configuration**
section of [The AI agent](#the-ai-agent-agent) below. The same `.env` file
configures both halves.

## Running the server

```bash
python server.py
```

It starts and waits silently — that's correct! It speaks MCP over stdio, so
it's meant to be launched by an MCP client, not used from a terminal
directly.

### Add it to Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config) and
add:

```json
{
  "mcpServers": {
    "mini-mcp-server": {
      "command": "/FULL/PATH/TO/mini-mcp-server/.venv/bin/python",
      "args": ["/FULL/PATH/TO/mini-mcp-server/server.py"],
      "env": { "GITHUB_TOKEN": "github_pat_..." }
    }
  }
}
```

Restart Claude Desktop, then ask: *"List my published Substack articles."*

## Testing with MCP Inspector

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) is the
official visual debugger for MCP servers — no AI client needed. It comes with
the SDK's CLI (installed via `requirements.txt`):

```bash
source .venv/bin/activate
mcp dev server.py
```

This opens the Inspector in your browser. Click **List Tools**, pick any
tool, fill in the arguments and hit *Run Tool* — try
`list_substack_articles` with `{"limit": 5}` first (it needs no token), then
`preview_article` with `{"article": "Week 11 Day 3"}`.

---

## The AI agent (`agent/`)

A small, readable agent that demonstrates the whole MCP loop end to end: an
LLM, connected through the official MCP SDK to an independently built server,
discovering its tools dynamically and using them to complete a multi-step
task - with a human in the loop for anything that writes.

No agent frameworks are involved: `requests` for the LLM APIs, the official
`mcp` SDK for MCP, `jsonschema` for argument validation. About 1,400 lines of
heavily commented teaching code across five files (plus ~1,000 lines of
tests).

### Quickstart

```bash
# 1. one-time setup (same venv as the server)
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 2. configure the LLM (see the table below) and your GitHub token
cp .env.example .env    # then edit .env

# 3. sanity-check tool discovery (needs no LLM key at all)
python -m agent --list-tools

# 4. talk to the agent
python -m agent
python -m agent "Find my latest Week 11 article and check if it is already in GitHub"

# 5. or in the browser: chat + live tool timeline + Approve/Deny buttons
python -m agent --web --demo     # scripted, no API key -> http://127.0.0.1:8765
python -m agent --web            # with your real LLM

# 6. or watch the whole example task with zero credentials in the terminal
python -m agent --demo
```

### Browser UI (`--web`)

Prefer a window to a terminal? The same agent, served locally:

```bash
python -m agent --web --demo   # try it now, no API key needed
python -m agent --web          # your real LLM
# -> http://127.0.0.1:8765 (Ctrl-C to stop; --port / --no-browser available)
```

What you get (one self-contained page, stdlib `http.server` only - no
frameworks, no new dependencies, no CDN assets):

- **Chat pane** with the conversation context (survives a page reload via
  `/api/history`).
- **Live tool timeline** streamed over Server-Sent Events: `⚙ name {args}`
  when a call starts, `✔ / ✖` with a result preview when it finishes.
- **Approve/Deny buttons** for dangerous tools - the same approval gate as
  the terminal y/N, now a click. The card shows the exact arguments and
  warns when the call can write to GitHub. New cards are highlighted and
  scrolled into view, and a banner above the composer keeps saying *"waiting
  for your decision"* until you click - typing "confirm" in the chat does
  **not** approve anything, only the button click does.
- **Tool sidebar** listing the discovered catalogue, flagging which tools
  require approval (`⚠ approval`) - straight from `tools/list`.

The UI is a thin transport: `agent/web.py` runs the existing `Agent` loop on
a background event loop and speaks HTTP/SSE to the page. Safeguards are
identical - human approval first, server-side `confirm`/duplicate checks
second - and if nobody clicks a decision within `AGENT_APPROVAL_TIMEOUT`
seconds (default 300) the call is **auto-denied** (silence is a "no"). It
binds to `127.0.0.1` only and rejects requests whose `Host` header is not
localhost; it is a local demo tool, not a public web app.

### LLM configuration (all via environment / `.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` (any OpenAI-compatible API), `anthropic`, or `mock` (offline) |
| `LLM_MODEL` | `gpt-4o-mini` / `claude-sonnet-4-5` | Model name sent to the provider (for `openai`, `OPENAI_MODEL` / `MODEL` are honoured as fallbacks) |
| `LLM_API_KEY` | – | The key; falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| `LLM_BASE_URL` | `https://api.openai.com/v1` (openai) / `https://api.anthropic.com` | Point it at OpenRouter, Ollama (`http://localhost:11434/v1`), LM Studio, ... (for `openai`, `OPENAI_BASE_URL` is honoured as a fallback) |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `LLM_MAX_RETRIES` | `2` | Retries on timeouts / 429 / 5xx (with backoff) |
| `LLM_MAX_TOKENS` | `8192` | Response budget (generous by default: reasoning-style models spend part of it on thinking) |

### Agent behaviour and limits

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_MAX_STEPS` | `10` | Hard cap on tool-loop iterations per user turn |
| `AGENT_MAX_TOOL_RESULT_CHARS` | `16000` | Tool results are truncated (marked) past this size |
| `AGENT_APPROVAL_TOOLS` | `publish_article_to_github` | Comma-separated tools that always require an explicit y/N from you. Tools advertising the MCP `destructiveHint` annotation are gated automatically too |
| `AGENT_SERVER_COMMAND` | this venv's `python server.py` | How the MCP server child process is launched |
| `AGENT_MOCK_SCRIPT` | – | JSON file of canned replies for `LLM_PROVIDER=mock` |
| `AGENT_APPROVAL_TIMEOUT` | `300` | Web UI only: seconds to wait for an Approve/Deny click before a dangerous call is auto-denied (fail closed) |
| `AGENT_LOG_LEVEL` / `AGENT_LOG_FILE` | `WARNING` / `agent.log` | Full audit trail (stderr too with `--verbose`) |

API keys are only ever sent as an HTTPS header to the endpoint you
configured, are never hardcoded, and are never written to `agent.log`.

### How tool discovery works (nothing is hardcoded)

At startup the agent spawns `python server.py` and sends a real `tools/list`
request. Each discovered tool (name, description, **input schema**) is kept
verbatim and handed to the LLM in its native tool format - `tools[].function`
for OpenAI, `tools[].input_schema` for Anthropic. The agent has no copy of
any tool definition: add a `@mcp.tool()` to `server.py` and the agent can use
it the next time it starts. `python -m agent --list-tools` shows the live
catalogue the model will see:

```
MCP server 'mini-mcp-server' online - 6 tools discovered:
  • list_substack_articles(search, limit, include_non_series_posts)
  • get_substack_article(article)  [required: article]
  • list_github_files(path, branch)
  • check_existing_article(article, file_path, branch)
  • preview_article(article, file_path, branch)  [required: article]
  • publish_article_to_github(article, confirm, file_path, branch, commit_message, overwrite)
```

### The execution loop

Per user message, the agent repeats until the model answers without tool
calls (or a limit fires):

1. **Understand** - the LLM sees the conversation + discovered tools.
2. **Select** - it asks for one or more tool calls.
3. **Generate arguments** - validated against the tool's discovered JSON
   schema *before* execution. Common scalar slips from small models
   (`limit:"2"` instead of `2`) are repaired in place; everything else is
   not. Invalid arguments never reach the server; the model receives the
   exact schema errors and self-corrects. Unknown tool names get back the
   list of tools that actually exist.
4. **Human approval** - if the tool is dangerous (see `AGENT_APPROVAL_TOOLS`
   / `destructiveHint`), the call pauses and you get a y/N prompt showing
   the arguments. Denied calls are never sent; the model is told the user
   declined and must change course.
5. **Execute** - `tools/call` over MCP; results are truncated if enormous
   and appended to the conversation.
6. **Inspect & decide** - the model reads the result and either continues or
   returns a final answer (or asks you a clarifying question).
7. **Stop** - on the final answer, on `AGENT_MAX_STEPS`, or when a result
   shows it cannot proceed.

### Approval flow - two independent layers

Publishing is protected twice, and **neither layer can be used to bypass the
other**:

- **Layer 1, the agent:** `publish_article_to_github` (and anything marked
  destructive) only runs after you type `y` at a prompt that shows the exact
  arguments. If you decline - or no human is available - the call is denied
  and nothing is sent to the server.
- **Layer 2, the server (unchanged):** `publish_article_to_github` still
  refuses unless `confirm=true`, re-reads the destination right before
  writing, refuses overwrites unless `overwrite=true`, forwards the file sha
  so GitHub itself rejects stale overwrites, and validates the path.

When - and only when - you approved a call whose arguments forgot
`confirm=true`, the agent sets it for you (the human decision is already
recorded). The server's checks then still run in full. The agent has no way
to write to GitHub except through this tool.

### Example: the Week 11 task, start to finish

```
you ▸ Find my latest Week 11 Substack article, check whether it exists in
      GitHub, prepare a Markdown preview, and publish it once I approve.
  ⚙ calling list_substack_articles {"search": "Week 11", "limit": 5}
  ✔ list_substack_articles -> {"success": true, "count": 5, ...}
  ⚙ calling get_substack_article {"article": "Week 11 Day 3"}
  ✔ get_substack_article -> {"success": true, "article": {"title": "Week 11, Day 3 - MCP...
  ⚙ calling check_existing_article {"article": "Week 11 Day 3"}
  ✔ check_existing_article -> {"exists": false, "existing_files_in_folder": [...]}
  ⚙ calling preview_article {"article": "Week 11 Day 3"}
  ✔ preview_article -> {"proposed_file_path": "11-week-11/day-3-mcp-how-ai-agents-connect.md", ...}

  ⚠ publish_article_to_github wants to run and needs your approval:
      arguments: {"article": "Week 11 Day 3", "confirm": true}
      This can WRITE to GitHub (a preview step should have shown the exact
      file path and content first).
    Approve this tool call? [y/N]:

agent ▸ I found "Week 11, Day 3 - MCP: How AI Agents Connect..." (published
        Nov 6). It does not exist in the repo yet... Shall I publish it to
        11-week-11/day-3-mcp-how-ai-agents-connect.md?
```

Typing `y` runs the publish tool; the result (file URL + commit URL) is
reported verbatim. Typing `n` stops the pipeline and the agent explains what
it had prepared. `--demo` runs this exact scenario offline with the mock
provider - the only difference from a real LLM is who writes the tool calls;
discovery, validation, approval and the server's safeguards are identical.

### Running the tests

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

What the suite covers (all offline unless noted):

- **Agent loop** - arguments validated against discovered schemas (bad
  arguments bounce back to the model with the exact errors), unknown tools
  rejected without touching the server, approval granted/denied paths,
  fail-closed when no human is reachable, `destructiveHint` handling,
  step-limit stop, result truncation, conversation persistence.
- **LLM providers** - OpenAI and Anthropic wire formats, tool-call parsing
  (including malformed arguments), retry-on-429, friendly 401/404 errors.
- **Web UI** - real HTTP on an ephemeral port: page/info/history endpoints,
  non-localhost `Host` rejected, Approve click → publish runs (with `confirm`
  injected after approval), Deny click → the call never reaches the server,
  and **no click within the timeout → auto-denied** (fail closed).
- **Real server integration (in-memory)** - the agent drives the *actual*
  `server.py` tool code with only the network stubbed: full publish after
  approval writes exactly one file at the repo's `NN-week-NN/day-D-<slug>.md`
  path; a denied approval writes nothing; **with the agent gate disabled** the
  server still refuses unconfirmed publishes, refuses overwriting without
  `overwrite=true`, forwards the sha when it is approved, and refuses bad
  paths (`../evil.py`).
- **Real stdio integration** - spawns `python server.py` as a child process
  and verifies discovery + a live public `list_substack_articles` call
  (skips cleanly when Substack is unreachable).

### Design notes (agent)

- **Server and agent stay separate.** The agent imports nothing from
  `server.py` in production - it only speaks MCP. The tests use an in-memory
  MCP connection purely to avoid the network, not to shortcut the protocol.
- **Single agent, single loop.** No multi-agent orchestration, no vector
  stores, no frameworks - the interesting logic fits in `agent/agent.py`.
- **The LLM is swappable.** One normalised message format; adding a provider
  means writing one class in `agent/llm.py`.
- **Graceful failures.** LLM 401/404/429, server errors and oversized
  results all surface as readable messages; the REPL survives them.
- **Honesty rule in the system prompt:** the model may not claim anything
  was published unless a tool result says so.

## Example user interactions

*(Through Claude Desktop or any MCP client; the
[agent CLI transcript](#example-the-week-11-task-start-to-finish) above shows
the same flow from your terminal.)*

**You:** *List my published articles from the 16-week AI/ML series.*
**Assistant:** calls `list_substack_articles` → shows the newest posts with
week/day numbers ("Week 11, Day 3 — MCP: How AI Agents Connect..." etc.).

**You:** *Fetch Week 11 Day 4.*
**Assistant:** calls `get_substack_article(article="Week 11 Day 4")` →
renders the Markdown for you. (If Day 4 isn't published yet, the tool says so
and shows what exists.)

**You:** *Add this article to GitHub.*
**Assistant:** calls `list_github_files` to inspect the repo,
`check_existing_article` to make sure it's new, and `preview_article` — then
shows you the Markdown and destination `11-week-11/day-4-<slug>.md` and asks
*"Shall I publish this?"*

**You:** *Confirm.*
**Assistant:** calls `publish_article_to_github(article=..., confirm=True)`
→ returns the new file link, e.g.
`https://github.com/niharika2309/16-week-ai-ml-series/blob/main/11-week-11/day-4-....md`.
It also reminds you that running `cd tools && npm install && node
build_book.mjs` regenerates the repo's book navigation.

If you ever say "replace the existing file", the assistant must additionally
pass `overwrite=True` — without it the tool refuses and changes nothing.

## Design notes & guarantees

- **Official SDK, current API.** Built on the official MCP Python SDK
  (`mcp` ≥ 2.x) using `from mcp.server import MCPServer` — the current name
  of the class historically known as `FastMCP`. A compatibility fallback for
  the v1 import (`mcp.server.fastmcp.FastMCP`) is included, so older
  environments keep working.
- **No new repos, no unrelated edits.** The server only ever creates/updates
  the single file you approved, at a path you saw in the preview.
- **No paywall bypass.** Subscriber-only posts are published as the free
  public preview with a 🔒 banner linking to the full post — matching the
  repo's existing convention.
- **Graceful errors.** Missing/expired token, unknown slug, GitHub rate
  limits, deleted articles and non-Markdown paths all come back as clear
  explanations the assistant can relay to you — never stack traces.
- **Duplicate safety.** `check_existing_article` + the publish-time re-read
  (with GitHub's sha requirement) mean the server cannot silently clobber a
  file that changed since the preview.
- **Two independent approval layers.** The agent asks a human before any
  dangerous tool runs; the server independently re-checks `confirm`,
  duplicates and paths. Tests prove each layer still holds when the other is
  removed — see [Approval flow](#approval-flow--two-independent-layers).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GITHUB_TOKEN is not set` | Copy `.env.example` to `.env`, add your token (or pass `GITHUB_TOKEN` in the MCP client's `env` block). |
| `GitHub rejected the token (401)` | Token expired/invalid or lacks *Contents: Read and write* on the repo — regenerate it. |
| `rate limit ... (403)` | Too many anonymous calls. A valid token raises the limit dramatically. |
| Article not in the listing | The RSS feed only carries the ~20 most recent posts — pass the article's URL or slug directly instead. |
| Nothing happens when running `python server.py` | Correct behaviour — stdio servers wait for a client. Use `mcp dev server.py` to interact. |
| `ModuleNotFoundError: mcp` | Activate the venv (`source .venv/bin/activate`) or use the venv's `python`/`mcp` binaries. |
| Agent: `No API key found for provider ...` | Set `LLM_API_KEY` (or `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`) in `.env`, or run with `LLM_PROVIDER=mock` / `--demo` to try the agent key-free. |
| Agent: `The LLM provider rejected the API key (401)` | The key is wrong/expired for the provider you selected with `LLM_PROVIDER`. |
| Agent: `HTTP 404 ... LLM_MODEL` | Model name or `LLM_BASE_URL` is wrong (e.g. pointing OpenAI mode at an Anthropic key). |
| Agent: `Could not start the MCP server` | Run `python -m agent --list-tools`; check `AGENT_SERVER_COMMAND` and that the venv has `requirements.txt` installed. |
| Agent: `I stopped after N tool steps` | The loop limit (`AGENT_MAX_STEPS`, default 10) fired — narrow the request or raise the limit. |
| Web UI: card says *auto-denied* | Nobody clicked Approve within `AGENT_APPROVAL_TIMEOUT` (default 300s) — fail-closed by design; a chat message like "confirm" does not approve, only the button does. Ask again and click Approve on the highlighted card. |
| Web UI: port already in use | `python -m agent --web` auto-tries the next free port; check the printed URL or pass `--port`. |
