"""
substack_reader.py
==================

Everything about *reading* from the public Substack, in one place.

Two data sources are used, both completely public (no login, no API key):

1. The RSS feed  - https://niharikabatra.substack.com/feed
   Gives us the list of published posts: title, date, URL, description.
   This is how `list_substack_articles` discovers the 16-week series posts
   and filters out unrelated announcement posts.

2. The public post data - https://niharikabatra.substack.com/api/v1/posts/<slug>
   Gives us the full article content as clean HTML (`body_html`), the
   publish date, and whether the post is free or subscriber-only.
   If that ever fails, we fall back to reading the public article page
   (and finally the feed's embedded copy of the content).

A subscriber-only post only exposes its free preview publicly - we never
try to bypass a paywall. The `is_preview` flag tells the caller (and the
user) that the content is partial.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from typing import Optional
from xml.etree.ElementTree import fromstring as xml_fromstring

import requests
from bs4 import BeautifulSoup

from article_parser import extract_week_and_day

# ---------------------------------------------------------------------------
# Configuration (public URLs only - nothing secret here)
# ---------------------------------------------------------------------------
SUBSTACK_BASE_URL = "https://niharikabatra.substack.com"
FEED_URL = f"{SUBSTACK_BASE_URL}/feed"
POSTS_API_URL = f"{SUBSTACK_BASE_URL}/api/v1/posts"

# A normal browser User-Agent: Substack (like most sites) is friendlier to it.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
        "mini-mcp-server/1.0"
    )
}
REQUEST_TIMEOUT_SECONDS = 30

# An article belongs to the 16-week series when its title (or slug) contains
# something like "Week 11, Day 3". Anything else ("Introducing my newsletter",
# product announcements, ...) is filtered out of the series listing.
SERIES_TITLE_PATTERN = re.compile(r"week[\s_-]*\d+[\s_,.-]*day[\s_-]*\d+", re.IGNORECASE)


class SubstackError(Exception):
    """Raised when something goes wrong talking to Substack.

    The message is written for humans - the MCP tools return it directly so
    the AI assistant can explain the problem to you.
    """


@dataclass
class ArticleSummary:
    """The 'card' for one article in a listing (no full body)."""

    title: str
    url: str
    slug: str
    published_date: str
    description: str
    week: Optional[int]
    day: Optional[int]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def slug_from_url(url_or_slug: str) -> str:
    """Accept either a full URL or a bare slug and return the slug.

    'https://niharikabatra.substack.com/p/my-article' -> 'my-article'
    'my-article'                                      -> 'my-article'
    """
    text = (url_or_slug or "").strip()
    if not text:
        raise SubstackError("Please provide a Substack article URL or slug.")
    if "/p/" in text:
        text = text.split("/p/", 1)[1]
    text = text.split("?")[0].split("#")[0].strip("/")
    if not text:
        raise SubstackError(f"Could not find an article slug in '{url_or_slug}'.")
    return text


def _strip_html_tags(html_fragment: str) -> str:
    """Turn a piece of HTML into plain text (used for feed descriptions)."""
    if not html_fragment:
        return ""
    text = BeautifulSoup(html_fragment, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _text(item, tag_name: str) -> str:
    """Read the text of a direct child tag of an RSS <item>, or ''."""
    node = item.find(tag_name)
    return (node.text or "").strip() if node is not None and node.text else ""


# ---------------------------------------------------------------------------
# 1) Listing articles via the RSS feed
# ---------------------------------------------------------------------------
def _fetch_feed_items() -> list:
    """Download the RSS feed and return the raw <item> elements."""
    try:
        response = requests.get(FEED_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SubstackError(f"Could not reach the Substack RSS feed: {exc}") from exc

    try:
        root = xml_fromstring(response.content)
    except Exception as exc:  # not valid XML - e.g. Substack returned an error page
        raise SubstackError("The Substack feed did not contain valid RSS/XML.") from exc

    return root.findall(".//item")


def _parse_feed_item(item) -> ArticleSummary:
    """Convert one RSS <item> into an ArticleSummary."""
    title = _text(item, "title")
    url = _text(item, "link")
    slug = slug_from_url(url) if url else ""
    week, day = extract_week_and_day(title=title, slug=slug)
    return ArticleSummary(
        title=title,
        url=url,
        slug=slug,
        published_date=_text(item, "pubDate"),
        description=_strip_html_tags(_text(item, "description")),
        week=week,
        day=day,
    )


def list_substack_articles(
    series_only: bool = True,
    search: Optional[str] = None,
    limit: Optional[int] = None,
) -> list:
    """Return the published articles as a list of plain dictionaries.

    - series_only=True keeps only posts that look like "Week N, Day N"
      entries of the 16-week AI/ML series (unrelated announcements dropped).
    - search filters by keyword in the title or description.
    - limit caps how many articles are returned (newest first).

    Note: an RSS feed only carries the most recent posts (Substack keeps 20).
    Articles you cannot find here can still be fetched directly by URL/slug.
    """
    articles = [_parse_feed_item(item) for item in _fetch_feed_items()]

    if series_only:
        articles = [a for a in articles if SERIES_TITLE_PATTERN.search(a.title or "")]

    if search:
        needle = search.lower().strip()
        articles = [
            a for a in articles
            if needle in a.title.lower() or needle in a.description.lower()
        ]

    # Newest first, using the RSS publication date.
    def sort_key(article: ArticleSummary):
        try:
            return parsedate_to_datetime(article.published_date)
        except (TypeError, ValueError):
            return None

    articles.sort(key=lambda a: sort_key(a) or 0, reverse=True)

    if limit is not None and limit > 0:
        articles = articles[:limit]

    return [asdict(a) for a in articles]


# ---------------------------------------------------------------------------
# 2) Fetching one article's full content
# ---------------------------------------------------------------------------
def fetch_substack_post(url_or_slug: str) -> dict:
    """Fetch one article and return everything the other modules need.

    Returned dictionary:
        title, slug, url, description, published_date,
        audience        - 'everyone' for free posts
        is_preview      - True when the post is subscriber-only, so the body
                          we received is just the public free preview
        body_html       - the article content as HTML
        week, day       - series numbers parsed from the title/slug (or None)

    Tries, in order: the public posts API, the public article page, the RSS
    feed's embedded content. Never any login or paywall bypass.
    """
    slug = slug_from_url(url_or_slug)

    post = _fetch_from_posts_api(slug) or _fetch_from_article_page(slug) or _fetch_from_feed(slug)

    if post is None:
        raise SubstackError(
            f"Could not find a published article with slug '{slug}'. "
            "Check the URL, or run list_substack_articles to see what is published."
        )

    title = post.get("title") or slug
    week, day = extract_week_and_day(title=title, slug=slug)
    audience = post.get("audience") or "everyone"

    return {
        "title": title,
        "slug": slug,
        "url": post.get("url") or f"{SUBSTACK_BASE_URL}/p/{slug}",
        "description": post.get("description") or "",
        "published_date": post.get("published_date") or "",
        "audience": audience,
        "is_preview": audience not in {"everyone", "free", None, ""},
        "body_html": post.get("body_html") or "",
        "week": week,
        "day": day,
    }


def _fetch_from_posts_api(slug: str) -> Optional[dict]:
    """Primary source: Substack's public post endpoint (no auth needed)."""
    try:
        response = requests.get(
            f"{POSTS_API_URL}/{slug}", headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS
        )
        if response.status_code != 200:
            return None
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    # Some Substack versions wrap the post under a "post" key; be tolerant.
    if isinstance(data, dict) and isinstance(data.get("post"), dict):
        data = data["post"]
    if not isinstance(data, dict) or not data.get("body_html"):
        return None

    return {
        "title": data.get("title") or slug,
        "url": data.get("canonical_url") or f"{SUBSTACK_BASE_URL}/p/{slug}",
        "description": _strip_html_tags(data.get("description") or ""),
        "published_date": data.get("post_date") or data.get("date") or "",
        "audience": data.get("audience") or "everyone",
        "body_html": data.get("body_html") or "",
    }


def _fetch_from_article_page(slug: str) -> Optional[dict]:
    """Fallback: read the public article page HTML and grab the article body."""
    url = f"{SUBSTACK_BASE_URL}/p/{slug}"
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
    except requests.RequestException:
        return None

    title = slug
    if soup.title and soup.title.string:
        title = soup.title.string.split("|")[0].strip() or slug

    # Try the usual Substack containers; keep the richest one that has content.
    candidates = soup.select("div.article-body, div.post-article, article")
    for candidate in candidates:
        if len(candidate.get_text(strip=True)) > 200:
            return {
                "title": title,
                "url": url,
                "description": "",
                "published_date": "",
                "audience": "everyone",
                "body_html": str(candidate),
            }
    return None


def _fetch_from_feed(slug: str) -> Optional[dict]:
    """Last resort: the feed itself carries the content of recent free posts."""
    try:
        items = _fetch_feed_items()
    except SubstackError:
        return None

    for item in items:
        link = _text(item, "link")
        if not link or slug_from_url(link) != slug:
            continue
        # content:encoded holds the full HTML body in Substack feeds.
        body_node = (
            item.find("encoded")
            or item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
            or item.find("description")
        )
        body_html = (body_node.text or "") if body_node is not None else ""
        if not body_html.strip():
            continue
        return {
            "title": _text(item, "title") or slug,
            "url": link,
            "description": _strip_html_tags(_text(item, "description")),
            "published_date": _text(item, "pubDate"),
            "audience": "everyone",
            "body_html": body_html,
        }
    return None
