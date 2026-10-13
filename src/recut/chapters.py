"""YouTube chapters: the model proposes chapter starts by sentence number; code enforces YouTube's rules (first
chapter at 00:00, at least three chapters, each at least 10 seconds) and formats them.

The offline baseline is a TextTiling-style lexical-cohesion segmenter: a chapter boundary where the vocabulary of
the sentences before and after a point overlaps least."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from recut.clips.suggest import transcript_lines
from recut.config import Settings
from recut.llm.base import ChatModel, LLMError, Message
from recut.llm.jsonparse import JSONReplyError, parse_json_object
from recut.logging_config import get_logger
from recut.models import Chapter, Sentence
from recut.text import ts

log = get_logger(__name__)

YOUTUBE_MIN_CHAPTER = 10.0
MIN_CHAPTER = 45.0
"""Our own floor: a chapter shorter than this is merged (YouTube's floor of 10 s is too permissive to be useful)."""
TITLE_MAX = 60

SYSTEM = """RECUT_TASK: chapters
You write YouTube chapters for a long video or podcast. Read the numbered transcript and mark where each new topic
or segment begins. Aim for about {k} chapters (roughly one every two minutes); a chapter starts where the speakers move to a
genuinely new subject, not at every new sentence. The first chapter starts at sentence {first}.
Titles: 2-7 words, specific to what is discussed (\"Why the first sensor failed\", not \"Discussion\"), no
numbering, no clickbait, max {title_max} characters.
Answer with JSON only: {{"chapters": [{{"start": <sentence number>, "title": "<title>"}}]}}"""

STOPWORDS = set(
    """a an the and or but so if then than that this these those to of in on at for with from by as is are was were be
    been being it its it's i you he she we they me him her us them my your our their what which who whom how why when
    where there here not no yes do does did have has had will would can could should may might just very really also
    about into over out up down more most some any all one two like know think going get got well um uh oh okay right
    yeah so's i'm you're we're they're that's there's don't didn't can't won't is it's let's lot thing things kind sort
    actually because said say says see much many even still now way back make made go""".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z'-]+", text.lower()) if t not in STOPWORDS and len(t) > 2]


def target_count(duration: float) -> int:
    """About one chapter every two minutes (typical for podcasts and talks on YouTube), 3 to 15."""
    return max(3, min(15, round(duration / 120)))


def validate(raw: Sequence[tuple[int, str]], sentences: Sequence[Sentence], *, duration: float) -> list[Chapter]:
    """Sort, dedupe, force the first chapter to 00:00, merge chapters shorter than MIN_CHAPTER, cap title length."""
    if not sentences:
        return []
    by_index = {s.index: s for s in sentences}
    seen: dict[int, str] = {}
    for idx, title in raw:
        if idx in by_index and idx not in seen:
            seen[idx] = title
    items = sorted(seen.items())
    if not items or items[0][0] != sentences[0].index:
        first_title = items[0][1] if items and by_index[items[0][0]].start < MIN_CHAPTER else "Introduction"
        if items and by_index[items[0][0]].start < MIN_CHAPTER:
            items = items[1:]
        items.insert(0, (sentences[0].index, first_title))
    chapters: list[Chapter] = []
    for idx, title in items:
        start = 0.0 if not chapters else math.floor(by_index[idx].start)
        title = _clean_title(title) or f"Part {len(chapters) + 1}"
        if chapters and start - chapters[-1].start < MIN_CHAPTER:
            continue
        chapters.append(Chapter(start=start, title=title, sentence=idx))
    if chapters and duration - chapters[-1].start < MIN_CHAPTER and len(chapters) > 1:
        chapters.pop()
    return chapters


def _clean_title(title: str) -> str:
    title = re.sub(r"^\s*(?:\d+[.):-]\s*|chapter \d+[:.-]?\s*)", "", str(title), flags=re.I)
    title = re.sub(r"\s+", " ", title).strip().strip('"').strip()
    if len(title) > TITLE_MAX:
        title = title[:TITLE_MAX].rsplit(" ", 1)[0].rstrip(",;:-")
    return title


def youtube_ready(chapters: Sequence[Chapter], duration: float) -> list[str]:
    """YouTube's published rules; empty list = valid."""
    problems = []
    if not chapters or chapters[0].start != 0:
        problems.append("the first chapter must start at 00:00")
    if len(chapters) < 3:
        problems.append("YouTube needs at least three chapters")
    for a, b in zip(chapters, [*chapters[1:], None], strict=False):
        end = b.start if b is not None else duration
        if end - a.start < YOUTUBE_MIN_CHAPTER:
            problems.append(f"chapter {a.title!r} is shorter than 10 seconds")
    return problems


def format_youtube(chapters: Sequence[Chapter]) -> str:
    return "\n".join(f"{ts(c.start)} {c.title}" for c in chapters)


def _compact(sentences: Sequence[Sentence], max_words: int = 24) -> list[Sentence]:
    """Long sentences shortened for the chapter prompt (topic boundaries do not need every word)."""
    out = []
    for s in sentences:
        words = s.text.split()
        text = s.text if len(words) <= max_words else " ".join(words[:max_words]) + " …"
        out.append(s.model_copy(update={"text": text}))
    return out


def make_chapters(
    sentences: Sequence[Sentence], model: ChatModel, settings: Settings, *, duration: float, title: str = ""
) -> tuple[list[Chapter], dict[str, Any]]:
    if not sentences:
        return [], {"calls": 0, "error": "empty transcript"}
    k = target_count(duration)
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM.format(k=k, first=sentences[0].index, title_max=TITLE_MAX)},
        {
            "role": "user",
            "content": (f"Video: {title}\n" if title else "")
            + f"Duration: {ts(duration)}\nTranscript, one sentence per line:\n"
            + transcript_lines(_compact(sentences), speakers=False),
        },
    ]
    try:
        completion = model.complete(messages, max_tokens=settings.llm_max_tokens, temperature=0.0)
        reply = parse_json_object(completion.text)
        raw = _parse(reply)
    except (LLMError, JSONReplyError) as exc:
        log.warning("chapters.model_failed", error=str(exc)[:200])
        raw = texttiling(sentences, k=k)
        chapters = validate(raw, sentences, duration=duration)
        return chapters, {"calls": 1, "fallback": "texttiling", "error": str(exc)[:200]}
    chapters = validate(raw, sentences, duration=duration)
    if len(chapters) < 3 and duration >= 180:
        # too few usable chapters from the model: fill with the lexical baseline's boundaries
        extra = texttiling(sentences, k=k)
        chapters = validate([*raw, *extra], sentences, duration=duration)
    return chapters, {"calls": 1, "raw": len(raw)}


def _parse(reply: dict[str, Any]) -> list[tuple[int, str]]:
    items = reply.get("chapters")
    if not isinstance(items, list):
        raise JSONReplyError("no chapters list in the reply")
    out = []
    for item in items:
        if isinstance(item, dict):
            try:
                out.append((int(item.get("start")), str(item.get("title") or "")))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return out


def texttiling(sentences: Sequence[Sentence], *, k: int, block: int = 6) -> list[tuple[int, str]]:
    """Lexical-cohesion boundaries (TextTiling, Hearst 1997, simplified): k-1 deepest similarity valleys between
    adjacent blocks of sentences, at least MIN_CHAPTER apart; titles from each chapter's most distinctive words."""
    n = len(sentences)
    if n == 0:
        return []
    toks = [_tokens(s.text) for s in sentences]
    if n < block * 3:
        return [(sentences[0].index, _keyword_title(toks, 0, n, toks))]
    sims = np.zeros(n)
    for gap in range(1, n):
        left = Counter(t for ts_ in toks[max(0, gap - block) : gap] for t in ts_)
        right = Counter(t for ts_ in toks[gap : gap + block] for t in ts_)
        common = set(left) & set(right)
        num = sum(left[t] * right[t] for t in common)
        den = math.sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values())) or 1.0
        sims[gap] = num / den
    smooth = np.convolve(sims, np.ones(3) / 3, mode="same")
    depth = np.zeros(n)
    for gap in range(1, n - 1):
        lpeak = smooth[max(1, gap - block * 2) : gap + 1].max()
        rpeak = smooth[gap : min(n, gap + block * 2)].max()
        depth[gap] = (lpeak - smooth[gap]) + (rpeak - smooth[gap])
    order = np.argsort(-depth)
    picked: list[int] = []
    for raw_gap in order:
        gap = int(raw_gap)
        if depth[gap] <= 0 or len(picked) >= k - 1:
            break
        t = sentences[gap].start
        if t < MIN_CHAPTER or sentences[-1].end - t < MIN_CHAPTER:
            continue
        if all(abs(t - sentences[p].start) >= MIN_CHAPTER * 2 for p in picked):
            picked.append(gap)
    bounds = [0, *sorted(picked)]
    out = []
    for i, b in enumerate(bounds):
        e = bounds[i + 1] if i + 1 < len(bounds) else n
        out.append((sentences[b].index, _keyword_title(toks, b, e, toks)))
    return out


def _keyword_title(toks: Sequence[list[str]], a: int, b: int, all_toks: Sequence[list[str]]) -> str:
    local = Counter(t for ts_ in toks[a:b] for t in ts_)
    df = Counter(t for ts_ in all_toks for t in set(ts_))
    n = max(1, len(all_toks))
    scored = sorted(local, key=lambda t: -(local[t] * math.log(1 + n / (1 + df[t]))))
    words = list(scored[:3])
    return " · ".join(w.capitalize() for w in words) if words else "Part"
