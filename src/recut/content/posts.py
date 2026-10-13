"""Long-form and social outputs from the transcript: SEO title, description and tags, show notes, a blog post and
one post per platform. Code enforces lengths, inserts chapters and the licence credit, renders quotes verbatim from
the transcript, and checks every number and quotation against what was said."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from recut.chapters import format_youtube
from recut.config import Settings
from recut.content import limits
from recut.content.grounding import TranscriptIndex, check
from recut.llm.base import ChatModel, LLMError, Message
from recut.llm.jsonparse import JSONReplyError, parse_json_object
from recut.logging_config import get_logger
from recut.models import Chapter, GroundingIssue, Outputs, Sentence, SocialPost, SourceInfo
from recut.text import ts

log = get_logger(__name__)

MAX_PROMPT_WORDS = 11000

METADATA_SYSTEM = """RECUT_TASK: metadata
You are the producer of a video podcast / YouTube channel. From the transcript, write the publishing metadata.
- "title": the YouTube title, max {title_target} characters, specific and honest (no clickbait the video does not
  deliver, no ALL CAPS).
- "description": 2-3 short paragraphs for the YouTube description. The first sentence must work on its own in
  search results. Do not add chapters, links or hashtags (they are added separately).
- "tags": 8-12 search tags, lowercase.
- "summary": one paragraph (60-90 words) for the show notes.
- "takeaways": 3-6 key takeaways, each one sentence.
- "quotes": the sentence numbers of 2-4 memorable, self-contained lines (they are quoted verbatim).
Only use facts, names and numbers that are in the transcript.
Answer with JSON only: {{"title": "", "description": "", "tags": [], "summary": "", "takeaways": [], "quotes": []}}"""

BLOG_SYSTEM = """RECUT_TASK: blog_post
You turn a video transcript into a blog post for the creator's website. Write {words} words of Markdown: a "# "
title, a two-sentence intro, 3-5 "## " sections that follow the video's structure, and a short conclusion. Write
as an article, not as a transcript ("In this episode..." only once at most). Keep the speaker's voice and
examples. Use only facts, names and numbers from the transcript; if you quote, quote exactly. No links, no
invented statistics. Answer with the Markdown only."""

SOCIAL_SYSTEM = """RECUT_TASK: social_posts
You write social media posts that promote a new video. One post per platform, each in that platform's style:
{specs}
Every post must stand on its own, lead with the most interesting idea from the video, and use only facts, names and
numbers from the transcript. Hashtags go at the end. No links (they are added when scheduling).
Answer with JSON only: {{{keys}}}"""


def plain_transcript(sentences: Sequence[Sentence], *, max_words: int = MAX_PROMPT_WORDS) -> str:
    """Paragraphs per speaker turn with a timestamp, numbered sentences; evenly thinned when too long."""
    total = sum(len(s.text.split()) for s in sentences)
    keep = list(sentences)
    if total > max_words:
        step = total / max_words
        keep = [s for i, s in enumerate(sentences) if int(i % step) == 0 or step < 1]
    lines = []
    for s in keep:
        lines.append(f"[{s.index}] {ts(s.start)} {s.speaker}: {s.text}")
    note = (
        "" if len(keep) == len(sentences) else f"(long transcript: {len(keep)} of {len(sentences)} sentences shown)\n"
    )
    return note + "\n".join(lines)


def _ask_json(model: ChatModel, messages: list[Message], settings: Settings) -> dict[str, Any]:
    completion = model.complete(messages, max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature)
    return parse_json_object(completion.text)


def _credit(source: SourceInfo) -> str:
    if source.attribution:
        return source.attribution
    if source.license and "public domain" not in source.license.lower():
        return f"Source: {source.title or source.filename} ({source.license})"
    return ""


def assemble_description(text: str, chapters: Sequence[Chapter], source: SourceInfo) -> str:
    parts = [text.strip()]
    if len(chapters) >= 3:
        parts.append("Chapters:\n" + format_youtube(chapters))
    credit = _credit(source)
    if credit:
        parts.append(credit)
    description = "\n\n".join(p for p in parts if p)
    return limits.truncate(description, limits.YOUTUBE_DESCRIPTION)


def _strings(value: Any) -> list[str]:
    return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []


def show_notes_markdown(
    summary: str,
    takeaways: Sequence[str],
    quote_ids: Sequence[int],
    sentences: Sequence[Sentence],
    chapters: Sequence[Chapter],
    credit: str,
) -> str:
    by_index = {s.index: s for s in sentences}
    out = ["## Summary", "", summary.strip(), ""]
    if takeaways:
        out += ["## Key takeaways", ""] + [f"- {t}" for t in takeaways] + [""]
    quotes = [by_index[i] for i in quote_ids if i in by_index][:4]
    if quotes:
        out += ["## Quotes", ""]
        for q in quotes:
            out += [f"> “{q.text.strip()}” ({ts(q.start)})", ""]
    if chapters:
        out += ["## Timestamps", ""] + [f"- {ts(c.start)} {c.title}" for c in chapters] + [""]
    if credit:
        out += ["---", "", credit, ""]
    return "\n".join(out).strip() + "\n"


def _ints(value: Any) -> list[int]:
    out = []
    if isinstance(value, list):
        for v in value:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
    return out


def _clean_tags(tags: Sequence[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        tag = re.sub(r"[#<>,\"]", "", tag).strip().lower()
        if tag and tag not in out and len(tag) <= 30:
            if limits.tags_total([*out, tag]) > limits.YOUTUBE_TAGS_TOTAL:
                break
            out.append(tag)
    return out[:15]


def generate(
    sentences: Sequence[Sentence],
    chapters: Sequence[Chapter],
    source: SourceInfo,
    model: ChatModel,
    settings: Settings,
    *,
    platforms: Sequence[str] = tuple(limits.PLATFORMS),
) -> tuple[Outputs, dict[str, Any]]:
    transcript = plain_transcript(sentences)
    index = TranscriptIndex(" ".join(s.text for s in sentences))
    stats: dict[str, Any] = {"calls": 0, "errors": []}
    header = f"Video: {source.title or source.filename}\nDuration: {ts(sentences[-1].end if sentences else 0)}\n"
    outputs = Outputs(generated_by=model.label)
    credit = _credit(source)

    # 1. metadata + show notes
    meta: dict[str, Any] = {}
    stats["calls"] += 1
    try:
        meta = _ask_json(
            model,
            [
                {"role": "system", "content": METADATA_SYSTEM.format(title_target=limits.YOUTUBE_TITLE_TARGET)},
                {"role": "user", "content": header + "Transcript:\n" + transcript},
            ],
            settings,
        )
    except (LLMError, JSONReplyError) as exc:
        stats["errors"].append(f"metadata: {str(exc)[:160]}")
    title = re.sub(r"\s+", " ", str(meta.get("title") or source.title or source.filename)).strip()
    outputs.seo_title = limits.truncate(title, limits.YOUTUBE_TITLE)
    outputs.seo_description = assemble_description(str(meta.get("description") or ""), chapters, source)
    outputs.tags = _clean_tags(_strings(meta.get("tags")))
    outputs.show_notes = show_notes_markdown(
        str(meta.get("summary") or ""),
        _strings(meta.get("takeaways"))[:6],
        _ints(meta.get("quotes")),
        sentences,
        chapters,
        credit,
    )

    # 2. blog post
    stats["calls"] += 1
    try:
        completion = model.complete(
            [
                {"role": "system", "content": BLOG_SYSTEM.format(words="700-1000")},
                {"role": "user", "content": header + "Transcript:\n" + transcript},
            ],
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        blog = re.sub(r"<think>.*?</think>", "", completion.text, flags=re.S).strip()
        blog = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", blog).strip()
        if credit:
            blog += f"\n\n---\n\n*{credit}*\n"
        outputs.blog_post = blog
    except LLMError as exc:
        stats["errors"].append(f"blog: {str(exc)[:160]}")

    # 3. social posts, with one retry for posts over the limit
    specs = "\n".join(
        f'- "{p.key}" ({p.name}): at most {p.target} characters, {p.hashtags} hashtag(s)'
        for p in (limits.PLATFORMS[k] for k in platforms)
    )
    keys = ", ".join(f'"{k}": ""' for k in platforms)
    social: dict[str, Any] = {}
    stats["calls"] += 1
    try:
        social = _ask_json(
            model,
            [
                {"role": "system", "content": SOCIAL_SYSTEM.format(specs=specs, keys=keys)},
                {"role": "user", "content": header + "Transcript:\n" + transcript},
            ],
            settings,
        )
    except (LLMError, JSONReplyError) as exc:
        stats["errors"].append(f"social: {str(exc)[:160]}")
    too_long = [
        k
        for k in platforms
        if limits.count(str(social.get(k) or ""), limits.PLATFORMS[k].counting) > limits.PLATFORMS[k].limit
    ]
    if too_long:
        stats["calls"] += 1
        try:
            fix = _ask_json(
                model,
                [
                    {"role": "system", "content": SOCIAL_SYSTEM.format(specs=specs, keys=keys)},
                    {"role": "user", "content": header + "Transcript:\n" + transcript},
                    {"role": "assistant", "content": str(social)[:6000]},
                    {
                        "role": "user",
                        "content": "These posts are over the platform limit; rewrite only them, shorter: "
                        + ", ".join(f"{k} (max {limits.PLATFORMS[k].limit} characters)" for k in too_long)
                        + ". Answer with JSON with only those keys.",
                    },
                ],
                settings,
            )
            for k in too_long:
                if isinstance(fix.get(k), str) and fix[k].strip():
                    social[k] = fix[k]
        except (LLMError, JSONReplyError) as exc:
            stats["errors"].append(f"social retry: {str(exc)[:160]}")
    for key in platforms:
        spec = limits.PLATFORMS[key]
        text = str(social.get(key) or "").strip()
        if not text:
            continue
        truncated = False
        if limits.count(text, spec.counting) > spec.limit:
            text = limits.truncate(text, spec.limit, spec.counting)
            truncated = True
        chars = limits.count(text, spec.counting)
        issues = check(text, index)
        if truncated:
            issues.append(GroundingIssue(kind="length", text="", message=f"cut to {spec.limit} characters by code"))
        outputs.social.append(
            SocialPost(
                platform=key,
                text=text,
                limit=spec.limit,
                chars=chars,
                within_limit=chars <= spec.limit,
                truncated=truncated,
                issues=issues,
            )
        )

    # grounding of the long-form texts
    long_issues: list[GroundingIssue] = []
    for label, text in (
        ("title", outputs.seo_title),
        ("description", str(meta.get("description") or "")),
        ("show notes", "\n".join([str(meta.get("summary") or ""), *_strings(meta.get("takeaways"))])),
        ("blog post", outputs.blog_post),
    ):
        for issue in check(text, index):
            long_issues.append(issue.model_copy(update={"message": f"{label}: {issue.message}"}))
    if len(title) > limits.YOUTUBE_TITLE_TARGET:
        long_issues.append(
            GroundingIssue(
                kind="length",
                text=title,
                message=f"title: {len(title)} characters; YouTube cuts titles in search after about {limits.YOUTUBE_TITLE_TARGET}",
            )
        )
    outputs.issues = long_issues
    return outputs, stats
