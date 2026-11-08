# mini-mcp-server

An [MCP](#what-is-mcp-in-plain-english) server that connects two halves of
Niharika's AI/ML education project:

- **Substack** — the [16-week AI/ML series](https://niharikabatra.substack.com/s/16-week-aiml-series) where the articles are published
- **GitHub** — the [16-week-ai-ml-series](https://github.com/niharika2309/16-week-ai-ml-series) curriculum repo where each article also lives as a Markdown file

With this server installed, an AI assistant (Claude Desktop, Cursor, ChatGPT
Desktop, ...) can do things like:

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

```
 AI assistant (Claude Desktop / Cursor / MCP Inspector)
        │
        │  MCP over stdio  (JSON-RPC on stdin/stdout)
        ▼
 ┌─────────────────────────────┐
 │ server.py                   │   @mcp.tool() x6  ← the MCP layer:
 │  (MCPServer, a.k.a. FastMCP)│   names, schemas, docstrings,
 │                             │   confirmation + duplicate guards
 └────────┬──────────┬─────────┘
          │          │
          ▼          ▼
 ┌────────────────┐  ┌────────────────┐        ┌─────────────────────────┐
 │ substack_reader│  │ github_writer  │──────▶ │ GitHub REST API          │
 │  .py           │  │  .py           │        │ api.github.com           │
 │  (requests)    │  │  (requests +   │        │ GITHUB_TOKEN (from .env) │
 └───────┬────────┘  │   Bearer token)│        └─────────────────────────┘
         ▼           └────────────────┘
 ┌────────────────┐
 │ article_parser │  HTML → clean Markdown, repo file-naming rules
 │  .py           │  (no magic library - the rules are readable code)
 └────────────────┘
```

| File | Responsibility |
|---|---|
| `server.py` | Creates the MCP server, defines the six tools, enforces the confirmation workflow. |
| `substack_reader.py` | Reads the public RSS feed (article discovery) and public post data (article content). Never touches private/paid content. |
| `article_parser.py` | Converts article HTML to Markdown (headings, lists, links, code blocks, quotes, images) and computes repo destination paths. |
| `github_writer.py` | All GitHub REST API calls: list the repo tree, read one file, create/update one file. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Template for your secret `GITHUB_TOKEN`. |

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

## Example user interactions

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

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GITHUB_TOKEN is not set` | Copy `.env.example` to `.env`, add your token (or pass `GITHUB_TOKEN` in the MCP client's `env` block). |
| `GitHub rejected the token (401)` | Token expired/invalid or lacks *Contents: Read and write* on the repo — regenerate it. |
| `rate limit ... (403)` | Too many anonymous calls. A valid token raises the limit dramatically. |
| Article not in the listing | The RSS feed only carries the ~20 most recent posts — pass the article's URL or slug directly instead. |
| Nothing happens when running `python server.py` | Correct behaviour — stdio servers wait for a client. Use `mcp dev server.py` to interact. |
| `ModuleNotFoundError: mcp` | Activate the venv (`source .venv/bin/activate`) or use the venv's `python`/`mcp` binaries. |
