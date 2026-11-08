"""
server.py
=========

The MCP server itself - the piece an AI assistant (Claude, ChatGPT, Cursor,
...) actually talks to.

How a tool becomes visible to an AI assistant, in 3 steps:

1. We create one server object (`MCPServer`, historically called `FastMCP`)
   and decorate plain Python functions with `@mcp.tool()`.
2. The SDK reads each function's name, type hints, defaults and docstring
   and turns them into the tool's JSON "input schema". When the assistant
   connects, it sends a `tools/list` request and receives that catalogue -
   this is how the assistant *discovers* what it can call.
3. The assistant later sends a `tools/call` request with arguments; the SDK
   runs our function and sends the returned JSON back as the tool result.

This server runs over the **stdio transport**: the assistant starts it as a
child process and they exchange JSON-RPC messages on stdin/stdout. That is
why NOTHING here may use plain `print()` - stdout is the protocol channel.
Use `_log()` below, which writes to stderr (safe for debug messages).

Run it manually with:  python server.py
Try it interactively:  mcp dev server.py   (opens MCP Inspector in a browser)
"""

from __future__ import annotations

import sys
from typing import Optional

from dotenv import load_dotenv

# Load GITHUB_TOKEN (and friends) from a .env file next to this script,
# if one exists. Real environment variables always win over .env values.
load_dotenv()

# --- MCP SDK import with a compatibility shim ------------------------------
# The official MCP Python SDK merged FastMCP into it as `FastMCP`; in SDK v2
# the same class was renamed `MCPServer`. We support both so the server works
# with `pip install "mcp>=2"` (current) and the older v1 line.
try:  # SDK v2 (current): from mcp.server import MCPServer
    from mcp.server import MCPServer as Server
except ImportError:  # SDK v1: from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp import FastMCP as Server

import article_parser
import github_writer
import substack_reader

# ---------------------------------------------------------------------------
# The server object. `instructions` is shown to the AI assistant when it
# connects, so it knows what this server is for and how to use it well.
# ---------------------------------------------------------------------------
mcp = Server(
    "mini-mcp-server",
    instructions=(
        "Connects Niharika's Substack '16-week AI/ML series' with its GitHub "
        "curriculum repo (niharika2309/16-week-ai-ml-series). "
        "Typical workflow: list_substack_articles -> get_substack_article -> "
        "check_existing_article -> preview_article -> ask the user for "
        "confirmation -> publish_article_to_github (confirm=True). "
        "Never publish without the user's explicit confirmation."
    ),
)


def _log(message: str) -> None:
    """Debug output that does NOT touch stdout (reserved for MCP messages)."""
    print(f"[mini-mcp-server] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Internal helpers (not tools - the assistant cannot call these directly)
# ---------------------------------------------------------------------------
_WEEK_DAY_REFERENCE = article_parser.WEEK_DAY_PATTERN


def _is_week_day_reference(text: str) -> bool:
    """True for human references like 'Week 11 Day 3'.

    Slugs and URLs always contain '-' or '/', so anything hyphen-free that
    matches 'week N day N' is treated as a human reference instead.
    """
    candidate = (text or "").strip().lower()
    if not candidate or "-" in candidate or "/" in candidate:
        return False
    return bool(_WEEK_DAY_REFERENCE.search(candidate))


def _resolve_article(article: str) -> dict:
    """Accept a URL, a slug, or a human reference like 'Week 11 Day 3'.

    Week/day references are looked up in the recent series listing (the RSS
    feed covers the newest posts); URLs and slugs are fetched directly.
    """
    text = (article or "").strip()
    if not text:
        raise substack_reader.SubstackError(
            "Please pass the article URL, its slug, or a reference like 'Week 11 Day 3'."
        )

    if _is_week_day_reference(text):
        match = _WEEK_DAY_REFERENCE.search(text)
        wanted_week, wanted_day = int(match.group(1)), int(match.group(2))
        for entry in substack_reader.list_substack_articles(series_only=True):
            if entry["week"] == wanted_week and entry["day"] == wanted_day:
                return substack_reader.fetch_substack_post(entry["url"])
        raise substack_reader.SubstackError(
            f"No article found for Week {wanted_week}, Day {wanted_day} in the "
            "recent feed. It may be older - pass its Substack URL or slug instead "
            "(see list_substack_articles)."
        )

    return substack_reader.fetch_substack_post(text)


def _article_markdown_and_path(article: dict, file_path: Optional[str]) -> tuple:
    """Build the final Markdown document and the destination path."""
    markdown = article_parser.build_markdown_document(article)

    if file_path:
        proposed = file_path.strip().strip("/")
    else:
        if article["week"] is None or article["day"] is None:
            raise ValueError(
                "This article does not look like a series post (no 'Week N, Day N'), "
                "so I cannot guess its path. Pass an explicit file_path instead."
            )
        proposed = article_parser.suggested_github_path(
            article["slug"], article["title"], article["week"], article["day"]
        )
    _validate_file_path(proposed)
    return markdown, proposed


def _validate_file_path(path: str) -> None:
    """Refuse anything that could escape the repo or clobber non-Markdown files."""
    if not path.lower().endswith(".md"):
        raise ValueError(f"File path must end with .md (got '{path}').")
    if path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"File path must be a plain path inside the repo (got '{path}').")


def _error(message: str) -> dict:
    """Uniform error payload so the assistant always gets a clear explanation."""
    return {"success": False, "error": message}


def _article_brief(article: dict) -> dict:
    """The public-facing metadata we return alongside article content."""
    return {
        "title": article["title"],
        "url": article["url"],
        "slug": article["slug"],
        "published_date": article["published_date"],
        "week": article["week"],
        "day": article["day"],
        "audience": article["audience"],
        "is_preview": article["is_preview"],
    }


# ===========================================================================
# TOOL 1 - Discover articles on Substack
# ===========================================================================
@mcp.tool()
def list_substack_articles(
    search: Optional[str] = None,
    limit: Optional[int] = None,
    include_non_series_posts: bool = False,
) -> dict:
    """List published articles from the 16-week AI/ML Substack series.

    Returns each article's title, publication date, URL, short description
    and its week/day numbers, newest first. Unrelated publication/announcement
    posts are filtered out unless include_non_series_posts is true.

    Args:
        search: Optional keyword to filter titles and descriptions (e.g. 'RAG').
        limit: Optional maximum number of articles to return.
        include_non_series_posts: Set true to also see non-series posts.
    """
    try:
        articles = substack_reader.list_substack_articles(
            series_only=not include_non_series_posts, search=search, limit=limit
        )
        return {
            "success": True,
            "count": len(articles),
            "note": "The RSS feed lists the most recent posts; older articles can still be fetched by URL or slug.",
            "articles": articles,
        }
    except substack_reader.SubstackError as exc:
        return _error(str(exc))


# ===========================================================================
# TOOL 2 - Read one article as Markdown
# ===========================================================================
@mcp.tool()
def get_substack_article(article: str) -> dict:
    """Fetch one Substack article and return its content as clean Markdown.

    Preserves headings, lists, links, code blocks, quotes and images. Great
    for 'show me Week 11 Day 3' style requests.

    Args:
        article: The article URL (https://niharikabatra.substack.com/p/...),
            its slug, or a reference like 'Week 11 Day 3'.
    """
    try:
        fetched = _resolve_article(article)
        markdown = article_parser.build_markdown_document(fetched)
        result = {
            "success": True,
            "article": _article_brief(fetched),
            "word_count": len(markdown.split()),
            "markdown": markdown,
        }
        if fetched["is_preview"]:
            result["warning"] = (
                "This post is subscriber-only on Substack, so this is only the "
                "free public preview - the full text is behind the paywall."
            )
        return result
    except (substack_reader.SubstackError, ValueError) as exc:
        return _error(str(exc))


# ===========================================================================
# TOOL 3 - Inspect the GitHub curriculum repo
# ===========================================================================
@mcp.tool()
def list_github_files(path: Optional[str] = None, branch: Optional[str] = None) -> dict:
    """List the files and folders of the GitHub curriculum repository.

    Shows the existing structure (01-week-1/, 02-week-2/, ...) so you can see
    what is already published before adding anything.

    Args:
        path: Optional folder filter, e.g. '09-week-9' to list one week only.
        branch: Branch name; defaults to the repo's main branch.
    """
    try:
        listing = github_writer.list_repository_files(
            branch=branch or github_writer.DEFAULT_BRANCH, path_prefix=path
        )
        listing["success"] = True
        if listing.get("truncated"):
            listing["warning"] = (
                "The repo is very large and the listing was truncated by GitHub; "
                "use the 'path' filter to inspect a specific folder."
            )
        return listing
    except github_writer.GitHubError as exc:
        return _error(str(exc))


# ===========================================================================
# TOOL 4 - Duplicate check
# ===========================================================================
@mcp.tool()
def check_existing_article(
    article: Optional[str] = None,
    file_path: Optional[str] = None,
    branch: Optional[str] = None,
) -> dict:
    """Check whether an article already exists in the GitHub repo.

    Prevents duplicate files and accidental overwrites. Give it either the
    article (URL/slug/'Week 11 Day 3') so the expected path is computed for
    you, or an exact file_path to check. It also lists any similar files in
    the same week folder (for example a file that exists under a slightly
    different name).

    Args:
        article: The Substack article URL, slug, or 'Week N Day N' reference.
        file_path: An exact repo path to check, e.g. '11-week-11/day-3-mcp.md'.
        branch: Branch name; defaults to the repo's main branch.
    """
    if not article and not file_path:
        return _error("Pass either 'article' or 'file_path' so I know what to check.")

    try:
        branch = branch or github_writer.DEFAULT_BRANCH

        if file_path:
            proposed = file_path.strip().strip("/")
            _validate_file_path(proposed)
            week_folder = proposed.rsplit("/", 1)[0] if "/" in proposed else None
        else:
            fetched = _resolve_article(article)
            if fetched["week"] is None or fetched["day"] is None:
                return _error(
                    "This article has no 'Week N, Day N' numbering, so I cannot "
                    "compute its expected path. Pass file_path instead."
                )
            proposed = article_parser.suggested_github_path(
                fetched["slug"], fetched["title"], fetched["week"], fetched["day"]
            )
            week_folder = f"{fetched['week']:02d}-week-{fetched['week']}"

        existing = github_writer.get_file(proposed, branch=branch)
        result = {
            "success": True,
            "repo": f"{github_writer.GITHUB_OWNER}/{github_writer.GITHUB_REPO}",
            "branch": branch,
            "file_path": proposed,
            "exists": existing is not None,
        }
        if existing:
            result["existing_file"] = {
                "sha": existing["sha"],
                "size": existing["size"],
                "url": existing["html_url"],
            }
            result["message"] = (
                "The file already exists. Publishing again would overwrite it - "
                "only do that with the user's explicit approval (overwrite=True)."
            )

        # Helpfulness: list sibling files in the same week folder so a near
        # duplicate under a different name is easy to spot.
        if week_folder:
            try:
                week_listing = github_writer.list_repository_files(
                    branch=branch, path_prefix=week_folder
                )
                result["existing_files_in_folder"] = [
                    f["path"] for f in week_listing["files"]
                ]
            except github_writer.GitHubError:
                pass  # sibling listing is a bonus; don't fail the whole check
        return result
    except (github_writer.GitHubError, substack_reader.SubstackError, ValueError) as exc:
        return _error(str(exc))


# ===========================================================================
# TOOL 5 - Preview before publishing
# ===========================================================================
@mcp.tool()
def preview_article(
    article: str,
    file_path: Optional[str] = None,
    branch: Optional[str] = None,
) -> dict:
    """Show exactly what WOULD be published: the Markdown and the destination.

    Shows the finished Markdown, the proposed path in the repo
    (NN-week-NN/day-D-<slug>.md following existing conventions), and whether
    that path already exists. Nothing is written to GitHub by this tool -
    always show this preview to the user and get their confirmation first.

    Args:
        article: The Substack article URL, slug, or 'Week N Day N' reference.
        file_path: Optional override for the automatically proposed path.
        branch: Branch name; defaults to the repo's main branch.
    """
    try:
        fetched = _resolve_article(article)
        markdown, proposed = _article_markdown_and_path(fetched, file_path)
        branch = branch or github_writer.DEFAULT_BRANCH

        existing = github_writer.get_file(proposed, branch=branch)

        result = {
            "success": True,
            "would_create_at": f"{github_writer.GITHUB_OWNER}/{github_writer.GITHUB_REPO}:{proposed}",
            "proposed_file_path": proposed,
            "branch": branch,
            "already_exists": existing is not None,
            "article": _article_brief(fetched),
            "markdown_length": len(markdown),
            "markdown": markdown,
        }
        if existing:
            result["warning"] = (
                "A file already exists at this path. Publishing would overwrite it "
                "and requires the user's explicit approval (overwrite=True)."
            )
        if fetched["is_preview"]:
            result["content_note"] = (
                "Subscriber-only post: this Markdown mirrors the free public preview."
            )
        return result
    except (substack_reader.SubstackError, github_writer.GitHubError, ValueError) as exc:
        return _error(str(exc))


# ===========================================================================
# TOOL 6 - Publish (only after the user confirms)
# ===========================================================================
@mcp.tool()
def publish_article_to_github(
    article: str,
    confirm: bool = False,
    file_path: Optional[str] = None,
    branch: Optional[str] = None,
    commit_message: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """Publish one Substack article to the GitHub repo as a Markdown file.

    SAFETY: this is the only tool that writes to GitHub. It refuses to run
    unless confirm=True, which must only be set after the user has seen the
    preview_article output and explicitly said yes. If a file already exists
    at the destination it refuses too, unless overwrite=True was explicitly
    approved by the user as well. It never touches any other file.

    Args:
        article: The Substack article URL, slug, or 'Week N Day N' reference.
        confirm: Must be true - the user explicitly confirmed the preview.
        file_path: Optional override for the proposed destination path.
        branch: Branch name; defaults to the repo's main branch.
        commit_message: Optional custom commit message.
        overwrite: Allow replacing an existing file (needs explicit approval).
    """
    if not confirm:
        return {
            "success": False,
            "published": False,
            "requires_confirmation": True,
            "message": (
                "Not published: I need explicit confirmation first. Please call "
                "preview_article, show the user the Markdown and destination path, "
                "and call me again with confirm=True only after they approve."
            ),
        }

    try:
        fetched = _resolve_article(article)
        markdown, proposed = _article_markdown_and_path(fetched, file_path)
        branch = branch or github_writer.DEFAULT_BRANCH

        # Duplicate protection: read the destination once; only reuse its sha
        # (which authorises a replace) when overwrite was explicitly approved.
        existing = github_writer.get_file(proposed, branch=branch)
        sha_to_send = None
        if existing:
            if not overwrite:
                return {
                    "success": False,
                    "published": False,
                    "file_already_exists": True,
                    "file_path": proposed,
                    "existing_file_url": existing["html_url"],
                    "message": (
                        f"'{proposed}' already exists, so nothing was changed. "
                        "Ask the user whether to replace it; if they approve, call "
                        "me again with overwrite=True (and confirm=True)."
                    ),
                }
            sha_to_send = existing["sha"]

        commit = commit_message or (
            f"Add Substack article: {fetched['title']}"
            if not overwrite or not existing
            else f"Update Substack article: {fetched['title']}"
        )

        _log(f"publishing {proposed} to {github_writer.GITHUB_OWNER}/{github_writer.GITHUB_REPO}@{branch}")
        created = github_writer.create_or_update_file(
            path=proposed,
            content=markdown,
            branch=branch,
            message=commit,
            sha=sha_to_send,
        )

        result = {
            "success": True,
            "published": True,
            "file_path": created["path"],
            "file_url": created["file_url"],
            "commit_url": created["commit_url"],
            "branch": created["branch"],
            "overwritten": bool(sha_to_send),
            "article": _article_brief(fetched),
            "message": (
                f"Published '{fetched['title']}' to {proposed}. "
                "Tip: run tools/build_book.mjs in the repo (cd tools && npm install && "
                "node build_book.mjs) to regenerate the book navigation."
            ),
        }
        if fetched["is_preview"]:
            result["content_note"] = (
                "Subscriber-only post: the file mirrors the free public preview and "
                "links to the full post on Substack, matching the repo's convention."
            )
        return result
    except (substack_reader.SubstackError, github_writer.GitHubError, ValueError) as exc:
        return _error(str(exc))


# ---------------------------------------------------------------------------
# Entry point.
# `transport="stdio"` means: run as a child process and speak MCP over
# stdin/stdout - exactly what Claude Desktop and the MCP Inspector do.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
