"""
github_writer.py
================

Everything about *talking to GitHub*, in one place.

Only the official GitHub REST API is used (https://api.github.com), via the
`requests` library. Your credentials come from the GITHUB_TOKEN environment
variable (loaded from your .env file) - the token is never hardcoded and
never sent anywhere except api.github.com.

Auth scopes you need: for the curriculum repo, a fine-grained token with
"Contents: Read and write" on niharika2309/16-week-ai-ml-series is enough.
Read-only tools (list/check) work with any token that can read the repo.

The three operations this module supports:

    list_repository_files()   -> GET  /repos/<owner>/<repo>/git/trees/<branch>
    get_file()                -> GET  /repos/<owner>/<repo>/contents/<path>
    create_or_update_file()   -> PUT  /repos/<owner>/<repo>/contents/<path>

Creating a file with the contents API also creates any missing parent
folders, so publishing `11-week-11/day-3-....md` creates `11-week-11/` too.
"""

from __future__ import annotations

import base64
import os
from typing import Optional
from urllib.parse import quote

import requests

# ---------------------------------------------------------------------------
# Configuration. Defaults point at the curriculum repo; the values can be
# overridden with environment variables (handy for testing) but you normally
# never need to change them.
# ---------------------------------------------------------------------------
GITHUB_API_BASE = os.environ.get("GITHUB_API_BASE_URL", "https://api.github.com").rstrip("/")
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "niharika2309")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "16-week-ai-ml-series")
DEFAULT_BRANCH = os.environ.get("GITHUB_DEFAULT_BRANCH", "main")

REQUEST_TIMEOUT_SECONDS = 30


class GitHubError(Exception):
    """A human-readable GitHub problem. The MCP tools return these messages
    so the AI assistant can tell you exactly what to fix (missing token,
    bad token, rate limit, ...)."""


def get_github_token() -> str:
    """Read the token from the environment. Never stored anywhere else."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise GitHubError(
            "GITHUB_TOKEN is not set. Add it to your .env file "
            "(copy .env.example to .env) or export it in your shell. "
            "Create a token at https://github.com/settings/tokens "
            "with permission to write contents to "
            f"{GITHUB_OWNER}/{GITHUB_REPO}."
        )
    return token


def _headers(token: str) -> dict:
    """Standard headers every GitHub REST call needs."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "mini-mcp-server",
        "Content-Type": "application/json",
    }


def _raise_helpful_error(response: requests.Response, context: str) -> None:
    """Turn a failed HTTP response into a GitHubError that explains itself."""
    status = response.status_code
    try:
        api_message = response.json().get("message", "")
    except ValueError:
        api_message = response.text[:200]

    if status == 401:
        message = (
            "GitHub rejected the token (401). GITHUB_TOKEN may be expired or "
            "invalid - generate a new one at https://github.com/settings/tokens."
        )
    elif status == 403:
        if "rate limit" in api_message.lower():
            message = (
                "GitHub's API rate limit was reached (403). Wait a bit or use a "
                "valid GITHUB_TOKEN - authenticated requests get a much higher limit."
            )
        else:
            message = (
                f"GitHub refused this action (403): {api_message}. Your token "
                f"probably lacks write access to {GITHUB_OWNER}/{GITHUB_REPO}."
            )
    elif status == 404:
        message = (
            f"GitHub returned 'not found' (404) for {context}. Check the repo "
            f"({GITHUB_OWNER}/{GITHUB_REPO}), branch and file path - or that "
            "your token can see this repository."
        )
    elif status == 409:
        message = (
            "GitHub reported a conflict (409): the file changed since it was "
            "last read. Run check_existing_article again and retry."
        )
    else:
        message = f"GitHub request for {context} failed ({status}): {api_message}"

    raise GitHubError(message)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------
def list_repository_files(
    branch: str = DEFAULT_BRANCH, path_prefix: Optional[str] = None
) -> dict:
    """Return every file/folder path in the repository (one API call).

    Uses the git "tree" endpoint, which lists the whole repo recursively.
    If path_prefix is given (e.g. '09-week-9'), only paths under it are kept.
    """
    token = get_github_token()
    url = f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/{quote(branch)}"
    response = requests.get(
        url, headers=_headers(token), params={"recursive": "1"}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        _raise_helpful_error(response, f"the file tree of branch '{branch}'")

    data = response.json()
    entries = data.get("tree", [])

    if path_prefix:
        prefix = path_prefix.strip().strip("/")
        entries = [e for e in entries if e.get("path", "").startswith(prefix + "/") or e.get("path") == prefix]

    files = [
        {"path": e.get("path"), "type": e.get("type"), "size": e.get("size")}
        for e in entries
        if e.get("type") == "blob"  # blobs are files; 'tree' entries are folders
    ]
    folders = sorted({e.get("path") for e in entries if e.get("type") == "tree"})

    return {
        "repo": f"{GITHUB_OWNER}/{GITHUB_REPO}",
        "branch": branch,
        "truncated": bool(data.get("truncated")),  # True = repo too large to list fully
        "file_count": len(files),
        "files": files,
        "folders": folders,
    }


def get_file(path: str, branch: str = DEFAULT_BRANCH) -> Optional[dict]:
    """Fetch one file's metadata + decoded text content.

    Returns None when the file does not exist (that is how duplicate
    checking works). Raises GitHubError for real problems.
    """
    token = get_github_token()
    url = f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{quote(path)}"
    response = requests.get(
        url, headers=_headers(token), params={"ref": branch}, timeout=REQUEST_TIMEOUT_SECONDS
    )

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        _raise_helpful_error(response, f"the file '{path}'")

    data = response.json()
    if not isinstance(data, dict) or "content" not in data:
        raise GitHubError(f"GitHub returned an unexpected response for '{path}'.")

    try:
        content_text = base64.b64decode(data["content"]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        content_text = None  # binary file - we only publish text files anyway

    return {
        "path": data.get("path", path),
        "sha": data.get("sha"),
        "size": data.get("size"),
        "html_url": data.get("html_url"),
        "download_url": data.get("download_url"),
        "content": content_text,
    }


# ---------------------------------------------------------------------------
# Write operation
# ---------------------------------------------------------------------------
def create_or_update_file(
    path: str,
    content: str,
    branch: str = DEFAULT_BRANCH,
    message: Optional[str] = None,
    sha: Optional[str] = None,
) -> dict:
    """Create (or, with a matching sha, update) one file in a single commit.

    - `sha` is required when *replacing* an existing file: GitHub demands the
      sha of the exact version you read, so we can never silently clobber a
      newer version someone else pushed. The caller (server.py) only passes
      this after the user explicitly approved an overwrite.
    - Missing parent folders are created automatically by GitHub.
    """
    token = get_github_token()
    commit_message = message or f"Add article: {path}"

    url = f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{quote(path)}"
    payload = {
        "message": commit_message,
        "branch": branch,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(url, headers=_headers(token), json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code not in {200, 201}:
        _raise_helpful_error(response, f"creating/updating '{path}'")

    data = response.json()
    content_info = data.get("content", {}) or {}
    commit_info = data.get("commit", {}) or {}

    return {
        "path": content_info.get("path", path),
        "sha": content_info.get("sha"),
        "file_url": content_info.get("html_url")
        or f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/blob/{branch}/{path}",
        "commit_url": commit_info.get("html_url"),
        "commit_sha": commit_info.get("sha"),
        "branch": branch,
    }
