"""Finding the best short-form moments: the model reads the numbered sentences and picks ranges; code validates,
snaps, lints and ranks them.

The model never gives timestamps: it answers with sentence numbers, so every clip starts and ends on a sentence that
exists. Long transcripts are read in overlapping windows; duplicates across windows are merged."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from recut.clips.lint import lint_clip, penalty
from recut.clips.snap import snap
from recut.config import Settings
from recut.llm.base import ChatModel, LLMError, Message
from recut.llm.jsonparse import JSONReplyError, parse_json_object
from recut.logging_config import get_logger
from recut.media.levels import Levels
from recut.models import Clip, ClipPost, Scores, Sentence, Word
from recut.text import ts

log = get_logger(__name__)

SYSTEM = """RECUT_TASK: suggest_clips
You are an experienced short-form video editor. From the numbered transcript of a long video or podcast, pick the
moments that work best as standalone vertical clips (TikTok, Instagram Reels, YouTube Shorts).

A good clip:
- is a contiguous range of sentences, {min_s:.0f} to {max_s:.0f} seconds long (use the timestamps; aim for 30-50 s);
- makes complete sense to someone who has NOT seen the rest of the video: no "as I said", no "this" or "he" pointing
  to something outside the clip, no unexplained names or acronyms;
- opens with a hook in the first sentence: a bold claim, a surprising number, a question that is answered in the
  clip, a strong opinion or the start of a story - not a greeting, filler or a transition;
- ends on the payoff: the point, the punchline, the answer or the lesson - not mid-thought and not on an open
  question;
- has one idea; concrete stories, numbers, tips, mistakes and contrarian opinions beat general talk.

Rules: use only sentence numbers that appear in the transcript; never invent content; clips must not overlap; the
"hook" field quotes the clip's opening words exactly. Scores are integers: hook, payoff, standalone and clarity
0-10, virality 0-100 (how likely it is to be watched to the end and shared). Be critical: a mediocre moment gets
5, not 8.

Answer with JSON only, no prose:
{{"clips": [{{"start": <first sentence number>, "end": <last sentence number>, "title": "<max 60 chars, accurate,
no clickbait>", "hook": "<the opening words>", "reason": "<one sentence: why it works>", "scores": {{"hook": 0,
"payoff": 0, "standalone": 0, "clarity": 0}}, "virality": 0, "caption": "<1-2 sentence post caption>",
"hashtags": ["#tag", "#tag"]}}]}}
Return at most {n} clips, best first."""


def transcript_lines(sentences: Sequence[Sentence], *, speakers: bool = True) -> str:
    """`[12] 03:21.4-03:27.9 (6.5 s) S1: text` per sentence: the format every clip and chapter prompt uses."""
    lines = []
    for s in sentences:
        who = f"{s.speaker}: " if speakers else ""
        lines.append(
            f"[{s.index}] {ts(s.start, millis=True)}-{ts(s.end, millis=True)} ({s.end - s.start:.1f} s) {who}{s.text}"
        )
    return "\n".join(lines)


LINE = re.compile(r"^\[(\d+)\] (\S+)-(\S+) \(([\d.]+) s\) (?:(S\d+): )?(.*)$")


def parse_lines(text: str) -> list[tuple[int, float, float, str, str]]:
    """Inverse of transcript_lines (used by the offline model): (index, start, end, speaker, text)."""
    out = []
    for line in text.splitlines():
        if m := LINE.match(line.strip()):
            out.append((int(m.group(1)), _parse_ts(m.group(2)), _parse_ts(m.group(3)), m.group(5) or "S1", m.group(6)))
    return out


def _parse_ts(value: str) -> float:
    parts = [float(p) for p in value.split(":")]
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


@dataclass
class Candidate:
    first: int
    last: int
    title: str
    hook: str
    reason: str
    scores: Scores
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)


@dataclass
class SuggestReport:
    clips: list[Clip]
    candidates: int
    rejected: list[str]
    windows: int
    model: str
    calls: int


def windows(sentences: Sequence[Sentence], *, minutes: float, overlap_minutes: float = 1.0) -> list[list[Sentence]]:
    if not sentences:
        return []
    total = sentences[-1].end - sentences[0].start
    if total <= minutes * 60 * 1.15:
        return [list(sentences)]
    out: list[list[Sentence]] = []
    start_t = sentences[0].start
    while True:
        end_t = start_t + minutes * 60
        chunk = [s for s in sentences if s.start >= start_t and s.start < end_t]
        if chunk:
            out.append(chunk)
        if end_t >= sentences[-1].end:
            break
        start_t = end_t - overlap_minutes * 60
    # a short tail window is merged into the previous one
    if len(out) >= 2 and out[-1][-1].end - out[-1][0].start < minutes * 60 * 0.35:
        tail = out.pop()
        seen = {s.index for s in out[-1]}
        out[-1].extend(s for s in tail if s.index not in seen)
    return out


def clips_wanted(duration: float, settings: Settings) -> int:
    return max(2, min(settings.max_clips, math.ceil(duration / 600 * settings.clips_per_10_minutes)))


def _as_float(value: Any, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, number))


def parse_candidates(reply: dict[str, Any], valid: set[int]) -> tuple[list[Candidate], list[str]]:
    out: list[Candidate] = []
    rejected: list[str] = []
    items = reply.get("clips")
    if not isinstance(items, list):
        return out, ["the reply has no clips list"]
    for item in items:
        if not isinstance(item, dict):
            rejected.append("a clip entry is not an object")
            continue
        try:
            first, last = int(item.get("start")), int(item.get("end"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            rejected.append(f"non-numeric sentence range {item.get('start')!r}-{item.get('end')!r}")
            continue
        if first > last:
            first, last = last, first
        if first not in valid or last not in valid:
            rejected.append(f"sentence range {first}-{last} is not in the transcript")
            continue
        raw_scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
        assert isinstance(raw_scores, dict)
        scores = Scores(
            hook=_as_float(raw_scores.get("hook"), 0, 10),
            payoff=_as_float(raw_scores.get("payoff"), 0, 10),
            standalone=_as_float(raw_scores.get("standalone"), 0, 10),
            clarity=_as_float(raw_scores.get("clarity"), 0, 10),
            virality=_as_float(item.get("virality"), 0, 100),
        )
        tags = item.get("hashtags")
        hashtags = [str(t) for t in tags if isinstance(t, str)] if isinstance(tags, list) else []
        out.append(
            Candidate(
                first=first,
                last=last,
                title=str(item.get("title") or "").strip(),
                hook=str(item.get("hook") or "").strip(),
                reason=str(item.get("reason") or "").strip(),
                scores=scores,
                caption=str(item.get("caption") or "").strip(),
                hashtags=hashtags,
            )
        )
    return out, rejected


def rank_score(scores: Scores, issue_penalty: float) -> float:
    model = (
        scores.hook * 2.5 + scores.payoff * 2.5 + scores.standalone * 3.0 + scores.clarity * 1.0 + scores.virality * 0.1
    )
    return round(max(0.0, min(100.0, model - issue_penalty)), 1)


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = min(a[1] - a[0], b[1] - b[0])
    return inter / shorter if shorter > 0 else 0.0


def _clean_title(title: str, fallback: str) -> str:
    title = re.sub(r"\s+", " ", title).strip().strip('"')
    if not title:
        title = fallback
    if len(title) > 60:
        cut = title[:60].rsplit(" ", 1)[0]
        title = cut.rstrip(",;:-") + "…"
    return title


def _grounded_hook(hook: str, opening: str) -> str:
    """The hook must be the clip's actual opening words; otherwise the first sentence is used."""
    if hook and fuzz.partial_ratio(hook.lower(), opening.lower()) >= 85:
        return hook
    return opening if len(opening) <= 140 else opening[:137].rsplit(" ", 1)[0] + "…"


def build_clips(
    candidates: Sequence[Candidate],
    sentences: Sequence[Sentence],
    words: Sequence[Word],
    levels: Levels | None,
    settings: Settings,
    *,
    source: str,
    media_end: float | None = None,
) -> tuple[list[Clip], list[str]]:
    rejected: list[str] = []
    built: list[Clip] = []
    for cand in candidates:
        result = snap(
            cand.first,
            cand.last,
            sentences,
            words,
            levels,
            min_s=settings.clip_min_seconds,
            max_s=settings.clip_max_seconds,
            pad_in=settings.pad_in_seconds,
            pad_out=settings.pad_out_seconds,
            media_end=media_end,
        )
        issues = lint_clip(
            result.first_sentence,
            result.last_sentence,
            sentences,
            words,
            start=result.start,
            end=result.end,
            min_s=settings.clip_min_seconds,
            max_s=settings.clip_max_seconds,
            cut_in=result.cut_in,
            cut_out=result.cut_out,
        )
        opening = sentences[result.first_sentence].text
        clip = Clip(
            id="clip_" + uuid.uuid4().hex[:8],
            start=result.start,
            end=result.end,
            first_sentence=result.first_sentence,
            last_sentence=result.last_sentence,
            title=_clean_title(cand.title, opening[:60]),
            hook=_grounded_hook(cand.hook, opening) if result.first_sentence == cand.first else opening,
            reason=cand.reason,
            scores=cand.scores,
            score=rank_score(cand.scores, penalty(issues)),
            cut_in=result.cut_in,
            cut_out=result.cut_out,
            issues=issues,
            adjustments=result.adjustments,
            post=ClipPost(caption=cand.caption, hashtags=cand.hashtags[:8]),
            source=source,
        )
        clip.title_card_text = clip.title
        built.append(clip)
    # dedupe overlapping picks (from overlapping windows or a model that repeats itself): keep the best
    built.sort(key=lambda c: c.score, reverse=True)
    kept: list[Clip] = []
    for clip in built:
        clash = next((k for k in kept if _overlap((clip.start, clip.end), (k.start, k.end)) > 0.3), None)
        if clash is not None:
            rejected.append(f"{ts(clip.start)}-{ts(clip.end)} overlaps a better clip ({clash.title!r})")
            continue
        kept.append(clip)
    for rank, clip in enumerate(kept, start=1):
        clip.rank = rank
    return kept, rejected


def suggest(
    sentences: Sequence[Sentence],
    words: Sequence[Word],
    levels: Levels | None,
    model: ChatModel,
    settings: Settings,
    *,
    title: str = "",
    media_end: float | None = None,
) -> SuggestReport:
    if not sentences:
        return SuggestReport([], 0, ["empty transcript"], 0, model.label, 0)
    duration = sentences[-1].end - sentences[0].start
    wanted = clips_wanted(duration, settings)
    parts = windows(sentences, minutes=settings.window_minutes)
    per_window = max(2, math.ceil(wanted / len(parts)) + 1)
    candidates: list[Candidate] = []
    rejected: list[str] = []
    calls = 0
    for part in parts:
        messages: list[Message] = [
            {
                "role": "system",
                "content": SYSTEM.format(
                    min_s=settings.clip_min_seconds, max_s=settings.clip_max_seconds, n=per_window
                ),
            },
            {
                "role": "user",
                "content": (f"Video: {title}\n" if title else "")
                + f"Transcript ({ts(part[0].start)}-{ts(part[-1].end)}), one sentence per line:\n"
                + transcript_lines(part),
            },
        ]
        calls += 1
        try:
            completion = model.complete(
                messages, max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature
            )
            reply = parse_json_object(completion.text)
        except (LLMError, JSONReplyError) as exc:
            log.warning("clips.window_failed", error=str(exc)[:200])
            rejected.append(f"window {ts(part[0].start)}-{ts(part[-1].end)}: {str(exc)[:160]}")
            continue
        found, bad = parse_candidates(reply, {s.index for s in part})
        candidates.extend(found)
        rejected.extend(bad)
    clips, dup = build_clips(
        candidates,
        sentences,
        words,
        levels,
        settings,
        source="rules" if model.is_local else "model",
        media_end=media_end,
    )
    clips = clips[: max(wanted, 1)]
    return SuggestReport(clips, len(candidates), rejected + dup, len(parts), model.label, calls)
