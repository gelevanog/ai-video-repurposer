"""Platform text limits and how each platform counts characters.

Limits are the platforms' published hard limits (checked October 2026). X counts every URL as 23 characters and
most emoji and CJK characters as 2; the others count Unicode code points."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Platform:
    key: str
    name: str
    limit: int
    """Hard limit; the post is cut (and flagged) above it."""
    target: int
    """What the prompt asks for: shorter than the limit reads better in the feed."""
    hashtags: int
    """How many hashtags the prompt asks for."""
    counting: str = "codepoints"


PLATFORMS: dict[str, Platform] = {
    "x": Platform("x", "X (Twitter)", 280, 260, 2, counting="x"),
    "linkedin": Platform("linkedin", "LinkedIn", 3000, 1300, 3),
    "instagram": Platform("instagram", "Instagram", 2200, 900, 5),
    "tiktok": Platform("tiktok", "TikTok", 4000, 600, 4),
    "threads": Platform("threads", "Threads", 500, 450, 1),
    "facebook": Platform("facebook", "Facebook", 63206, 900, 2),
}
YOUTUBE_TITLE = 100
YOUTUBE_TITLE_TARGET = 70
YOUTUBE_DESCRIPTION = 5000
YOUTUBE_TAGS_TOTAL = 500
SHORTS_CAPTION = 2200

_URL = re.compile(r"https?://\S+")


def _is_wide(ch: str) -> bool:
    if ord(ch) > 0xFFFF:
        return True  # emoji and other astral characters
    return unicodedata.east_asian_width(ch) in ("W", "F")


def count(text: str, counting: str = "codepoints") -> int:
    if counting != "x":
        return len(text)
    total = 0
    pos = 0
    for match in _URL.finditer(text):
        total += sum(2 if _is_wide(c) else 1 for c in text[pos : match.start()]) + 23
        pos = match.end()
    total += sum(2 if _is_wide(c) else 1 for c in text[pos:])
    return total


def truncate(text: str, limit: int, counting: str = "codepoints") -> str:
    """Cut at a word boundary so that the result, with an ellipsis, fits the limit."""
    if count(text, counting) <= limit:
        return text
    words = text.split(" ")
    out = ""
    for word in words:
        candidate = (out + " " + word) if out else word
        if count(candidate + "…", counting) > limit:
            break
        out = candidate
    return out.rstrip(",;:—- ") + "…"


def tags_total(tags: list[str]) -> int:
    """YouTube counts tags joined by commas; a tag with a space is quoted (+2)."""
    return sum(len(t) + (2 if " " in t else 0) for t in tags) + max(0, len(tags) - 1)
