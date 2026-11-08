"""
article_parser.py
=================

Turns the raw HTML of a Substack article into clean, readable Markdown.

Why this file exists
--------------------
Substack gives us article content as HTML (a big string of <p>, <h2>, <pre>
tags). GitHub stores our curriculum as Markdown (.md) files. This module is
the translator between the two: HTML in, Markdown out.

It also knows the *naming conventions* of the curriculum repository:

    <NN>-week-<NN>/day-<D>-<slug>.md
    e.g.  11-week-11/day-3-mcp-how-ai-agents-connect.md

so the AI assistant always proposes the same file layout you already use.

No external "markdown converter" library is used - the conversion rules are
written out below on purpose, so a beginner can read and tweak them.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Tags we never want to appear in the Markdown output.
# ---------------------------------------------------------------------------
DROP_TAGS = {"script", "style", "svg", "button", "form", "input", "link", "meta"}

# Block-level tags: each one becomes its own paragraph/section in Markdown.
BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote", "pre", "hr",
    "figure", "figcaption", "table", "tr", "section", "article", "main",
}


# ---------------------------------------------------------------------------
# Main entry point: HTML -> Markdown
# ---------------------------------------------------------------------------
def html_to_markdown(html: str) -> str:
    """Convert Substack article HTML into a clean Markdown string.

    Handles the things Substack articles actually contain:
    headings, paragraphs, bold/italic, links, inline code, fenced code
    blocks, bullet/numbered lists, blockquotes, images and horizontal rules.
    """
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Remove tags that never carry readable content.
    for tag in soup.find_all(list(DROP_TAGS)):
        tag.decompose()

    blocks = _render_children_as_blocks(soup)
    markdown = "\n\n".join(block for block in blocks if block.strip())
    return _cleanup(markdown)


# ---------------------------------------------------------------------------
# Block-level rendering
# ---------------------------------------------------------------------------
def _render_children_as_blocks(node: Tag) -> list:
    """Walk a node's children, group inline runs into paragraphs, and
    render real block elements recursively. Returns a list of Markdown blocks.
    """
    blocks: list = []
    inline_run: list = []  # consecutive inline nodes collected into one paragraph

    def flush_paragraph() -> None:
        if inline_run:
            text = "".join(_render_inline(child) for child in inline_run).strip()
            if text:
                blocks.append(text)
            inline_run.clear()

    for child in node.children:
        if isinstance(child, NavigableString):
            if str(child).strip():
                inline_run.append(child)
            continue
        if not isinstance(child, Tag):
            continue

        name = child.name.lower()
        if name in DROP_TAGS:
            continue

        if name in BLOCK_TAGS:
            flush_paragraph()
            rendered = _render_block(child)
            if rendered.strip():
                blocks.append(rendered)
        else:
            # <a>, <strong>, <em>, <code>, <img>, <span>, <br> ... are inline.
            inline_run.append(child)

    flush_paragraph()
    return blocks


def _render_block(tag: Tag) -> str:
    """Render one block-level tag to Markdown."""
    name = tag.name.lower()

    # Headings: <h2>Title</h2> -> "## Title"
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        text = _render_inline_children(tag).strip()
        return f"{'#' * level} {text}" if text else ""

    # Paragraphs (may contain images or inline code).
    if name == "p":
        return _render_inline_children(tag).strip()

    # Horizontal rule: <hr> -> "---"
    if name == "hr":
        return "---"

    # Code blocks: <pre><code class="language-python">...</code></pre>
    if name == "pre":
        return _render_code_block(tag)

    # Lists.
    if name in {"ul", "ol"}:
        return _render_list(tag, indent=0)

    # Blockquotes.
    if name == "blockquote":
        inner = "\n\n".join(_render_children_as_blocks(tag))
        quoted_lines = [
            f"> {line}" if line.strip() else ">" for line in inner.split("\n")
        ]
        return "\n".join(quoted_lines)

    # An image alone on a line.
    if name == "img":
        return _render_img(tag)

    # Tables: rendered simply as pipe-separated rows.
    if name == "table":
        return _render_table(tag)

    # <div>, <figure>, <section>, <figcaption> etc. are just containers:
    # render whatever is inside them.
    inner_blocks = _render_children_as_blocks(tag)
    return "\n\n".join(inner_blocks)


def _render_code_block(pre_tag: Tag) -> str:
    """Render <pre> as a fenced code block, keeping the language hint."""
    code_tag = pre_tag.find("code")

    # Substack marks language like: <code class="language-python">
    language = ""
    classes = (code_tag.get("class") or []) if code_tag else (pre_tag.get("class") or [])
    for cls in classes:
        if str(cls).startswith("language-"):
            language = str(cls)[len("language-"):]
    if language.lower() in {"plaintext", "text", "none", ""}:
        language = ""  # no useful language hint

    text = (code_tag or pre_tag).get_text()
    text = text.replace("\xa0", " ").strip("\n")

    # Use a fence longer than any backticks inside the code, so it never breaks.
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}"


def _render_list(list_tag: Tag, indent: int) -> str:
    """Render <ul>/<ol> (including nested lists) to Markdown."""
    ordered = list_tag.name.lower() == "ol"
    try:
        counter = int(list_tag.get("start", 1))
    except (TypeError, ValueError):
        counter = 1

    lines: list = []
    for li in list_tag.find_all("li", recursive=False):
        marker = f"{counter}. " if ordered else "- "
        counter += 1

        # Pull nested lists out first, render the item text, then the children.
        nested_lists = li.find_all(["ul", "ol"], recursive=False)
        li_copy = BeautifulSoup(str(li), "html.parser").find("li")
        for nested in li_copy.find_all(["ul", "ol"]):
            nested.decompose()
        item_text = _render_inline_children(li_copy).strip()

        indent_prefix = " " * indent
        lines.append(f"{indent_prefix}{marker}{item_text}")

        for nested in nested_lists:
            nested_md = _render_list(nested, indent + 2)
            if nested_md.strip():
                lines.append(nested_md)

    return "\n".join(lines)


def _render_table(table_tag: Tag) -> str:
    """Render a table as simple pipe rows (keeps it readable, not perfect)."""
    rows: list = []
    for tr in table_tag.find_all("tr"):
        cells = [_render_inline_children(cell).strip() for cell in tr.find_all(["td", "th"])]
        cells = [cell.replace("|", "\\|") for cell in cells]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _render_img(img_tag: Tag) -> str:
    """Render <img> as ![alt text](url)."""
    src = (img_tag.get("src") or img_tag.get("data-src") or "").strip()
    if not src:
        return ""
    alt = (img_tag.get("alt") or "article image").strip() or "article image"
    return f"![{alt}]({src})"


# ---------------------------------------------------------------------------
# Inline rendering (runs inside a paragraph, heading or list item)
# ---------------------------------------------------------------------------
def _render_inline_children(tag: Tag) -> str:
    return "".join(_render_inline(child) for child in tag.children)


def _render_inline(node) -> str:
    """Render an inline node (<a>, <strong>, <code>, <img>, text ...) to Markdown."""
    if isinstance(node, NavigableString):
        return str(node).replace("\xa0", " ")

    if not isinstance(node, Tag):
        return ""

    name = node.name.lower()

    if name in DROP_TAGS:
        return ""

    # <noscript> often contains <img> fallbacks Substack adds - keep the image,
    # skip everything else inside it.
    if name == "noscript":
        images = node.find_all("img")
        return "".join(_render_img(img) for img in images)

    if name == "br":
        return "\n"

    if name == "img":
        return _render_img(node)

    inner = _render_inline_children(node).strip()

    if name in {"strong", "b"}:
        return f"**{inner}**" if inner else ""

    if name in {"em", "i"}:
        return f"*{inner}*" if inner else ""

    if name in {"del", "s", "strike"}:
        return f"~~{inner}~~" if inner else ""

    if name == "code":
        if not inner:
            return ""
        # Avoid breaking the inline code span with backticks inside it.
        pad = " " if "`" in inner else ""
        return f"`{pad}{inner}{pad}`"

    if name == "a":
        href = (node.get("href") or "").strip()
        # Substack adds empty anchor links on headings - skip those.
        if not href or (not inner and href.startswith("#")):
            return inner
        label = inner or href
        return f"[{label}]({href})"

    # <span>, <u>, <sub>, <sup>, <font>, unknown tags: keep the text.
    return inner


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------
def _cleanup(markdown: str) -> str:
    """Tidy the generated Markdown: normalise spaces and blank lines."""
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in markdown.split("\n")]
    cleaned: list = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 1:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------------
# Article metadata helpers (titles, dates, file paths)
# ---------------------------------------------------------------------------
# Matches "Week 11, Day 3", "Week 11 Day 3", "week-11-day-3" etc.
WEEK_DAY_PATTERN = re.compile(r"week[\s_-]*(\d+)[\s_,.-]*day[\s_-]*(\d+)", re.IGNORECASE)


def extract_week_and_day(title: str = "", slug: str = "") -> Tuple[Optional[int], Optional[int]]:
    """Pull the (week, day) numbers out of an article title or URL slug."""
    for text in (title, slug):
        match = WEEK_DAY_PATTERN.search(text or "")
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def slugify(text: str) -> str:
    """Turn any string into a URL/file friendly slug (fallback when the
    Substack slug is missing): 'Hello, World!' -> 'hello-world'."""
    text = text.lower()
    text = re.sub(r"[&]", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def article_filename_slug(substack_slug: str, title: str, week: int, day: int) -> str:
    """Work out the file-name part of the article.

    The curriculum repo names files like the Substack URL slug *without* the
    leading 'week-11-day-3-' part:
        Substack slug 'week-11-day-3-mcp-how-ai-agents-connect'
        -> file slug   'mcp-how-ai-agents-connect'
    """
    base = (substack_slug or "").strip().strip("/")
    prefix_pattern = re.compile(rf"^week-?{week}-?day-?{day}-+", re.IGNORECASE)
    base = prefix_pattern.sub("", base)
    if not base:
        # Fall back to a slug built from the title.
        base = slugify(re.sub(WEEK_DAY_PATTERN, "", title or "", flags=re.IGNORECASE))
    return base


def suggested_github_path(substack_slug: str, title: str, week: int, day: int) -> str:
    """The full destination path inside the curriculum repo.

    Follows the existing repo layout: '<NN>-week-<NN>/day-<D>-<slug>.md'.
    """
    slug = article_filename_slug(substack_slug, title, week, day)
    return f"{week:02d}-week-{week}/day-{day}-{slug}.md"


def format_date(iso_date: Optional[str]) -> str:
    """'2026-11-06T23:58:05.605Z' -> 'November 6, 2026' (for the article header)."""
    if not iso_date:
        return ""
    try:
        parsed = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        if parsed.tzinfo:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%B %d, %Y").replace(" 0", " ")
    except ValueError:
        return iso_date.split("T")[0]


def build_markdown_document(article: dict) -> str:
    """Build the finished .md file content for one article.

    The header format matches the files already in the curriculum repo:

        # Title
        <blank>
        *November 2, 2026 · [Originally published on Substack](url)*
        <blank>
        (🔒 banner, only for subscriber-only posts)
        <blank>
        ...article body as Markdown...
    """
    title = article.get("title") or "Untitled article"
    url = article.get("url") or ""
    date_label = format_date(article.get("published_date"))

    header = f"# {title}\n\n*{date_label} · [Originally published on Substack]({url})*\n"

    banner = ""
    if article.get("is_preview"):
        banner = (
            "\n> 🔒 **This is a subscriber-only article on Substack.**\n"
            f"> This page mirrors the free preview — **[read the full article on "
            f"Substack →]({url})**\n"
        )

    body = html_to_markdown(article.get("body_html", ""))

    document = f"{header}\n{banner}\n{body}\n"
    return _cleanup(document) + "\n"
