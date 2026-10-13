"""Rule-based implementations of every model task, used by the offline `fake` provider (tests, CI, the zero-key
demo) and as the "rules only" baseline in the evaluation. They read the same prompts the real model reads."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from recut.chapters import _tokens, target_count, texttiling
from recut.clips.lint import CONNECTORS, CONTEXT_PHRASES, PRONOUN_STARTS
from recut.clips.suggest import parse_lines
from recut.llm.base import Message
from recut.llm.fake import FakeHandler
from recut.models import Sentence

HOOK_WORDS = re.compile(
    r"\b(truth|secret|mistake|nobody|never|always|biggest|worst|best|why|how|stop|wrong|surprising|lesson|rule|"
    r"trick|myth|actually|here is|here's|the problem|the reason|the one thing|most people)\b",
    re.I,
)
PAYOFF_WORDS = re.compile(
    r"\b(so the|that's why|that is why|the lesson|the answer|the point|which means|the result|in the end|it is a|"
    r"it's a|remember|the key|bottom line)\b",
    re.I,
)


def _user(messages: Sequence[Message]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


def _system(messages: Sequence[Message]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "system")


def _first_token(text: str) -> str:
    m = re.match(r"\W*([A-Za-z']+)", text)
    return m.group(1).lower() if m else ""


def _sentence_score(text: str) -> float:
    score = 0.0
    score += 2.0 * len(HOOK_WORDS.findall(text))
    score += 1.5 if "?" in text else 0.0
    score += 1.5 if re.search(r"\d", text) else 0.0
    score += 0.5 if re.search(r"\byou\b", text, re.I) else 0.0
    return score


def suggest_clips(messages: Sequence[Message]) -> str:
    lines = parse_lines(_user(messages))
    n = int(m.group(1)) if (m := re.search(r"Return at most (\d+) clips", _system(messages))) else 3
    lo = float(m.group(1)) if (m := re.search(r"(\d+) to (\d+) seconds long", _system(messages))) else 20.0
    hi = float(m.group(2)) if m else 60.0
    candidates = []
    for i, (_, start, _, _, text) in enumerate(lines):
        token = _first_token(text)
        if token in CONNECTORS or token in PRONOUN_STARTS:
            continue
        hook = _sentence_score(text)
        if hook <= 0:
            continue
        for j in range(i, len(lines)):
            end = lines[j][2]
            duration = end - start
            if duration > hi:
                break
            if duration < lo:
                continue
            body = " ".join(x[4] for x in lines[i : j + 1])
            last = lines[j][4].strip()
            ends_well = last.endswith((".", "!")) and not last.endswith("?")
            payoff = 2.0 * len(PAYOFF_WORDS.findall(body)) + (2.0 if ends_well else -3.0)
            context = -4.0 if CONTEXT_PHRASES.search(body) else 0.0
            ideal = -abs(duration - 38) / 10
            speakers = len({x[3] for x in lines[i : j + 1]})
            total = hook * 1.5 + payoff + context + ideal - (speakers - 1) * 0.7
            candidates.append((total, i, j, hook, payoff))
    candidates.sort(reverse=True)
    picked: list[tuple[float, int, int, float, float]] = []
    for cand in candidates:
        if all(cand[2] < p[1] or cand[1] > p[2] for p in picked):
            picked.append(cand)
        if len(picked) >= n:
            break
    clips = []
    for total, i, j, hook, payoff in picked:
        opening = lines[i][4]
        clips.append(
            {
                "start": lines[i][0],
                "end": lines[j][0],
                "title": _title_from(opening),
                "hook": " ".join(opening.split()[:12]),
                "reason": "rule-based pick: hook words or a question up front, ends on a full stop",
                "scores": {
                    "hook": round(min(10, 4 + hook), 1),
                    "payoff": round(min(10, max(0, 5 + payoff)), 1),
                    "standalone": 7,
                    "clarity": 7,
                },
                "virality": round(min(95, 40 + total * 3), 0),
                "caption": opening[:150],
                "hashtags": ["#podcast", "#shorts"],
            }
        )
    return json.dumps({"clips": clips})


def _title_from(text: str) -> str:
    words = re.sub(r"[^\w\s'’-]", "", text).split()
    title = " ".join(words[:8])
    return title[:1].upper() + title[1:] if title else "Clip"


def _sentences(messages: Sequence[Message]) -> list[Sentence]:
    out = []
    for idx, start, end, speaker, text in parse_lines(_user(messages)):
        out.append(Sentence(index=idx, start=start, end=end, text=text, speaker=speaker, first_word=0, last_word=0))
    return out


def chapters(messages: Sequence[Message]) -> str:
    sentences = _sentences(messages)
    if not sentences:
        return json.dumps({"chapters": []})
    k = target_count(sentences[-1].end)
    return json.dumps({"chapters": [{"start": i, "title": t} for i, t in texttiling(sentences, k=k)]})


_PLAIN = re.compile(r"^\[(\d+)\] (\S+) (S\d+): (.*)$")


def _plain(messages: Sequence[Message]) -> list[tuple[int, str]]:
    out = []
    for line in _user(messages).splitlines():
        if m := _PLAIN.match(line.strip()):
            out.append((int(m.group(1)), m.group(4)))
    return out


def _keywords(texts: Sequence[str], k: int = 10) -> list[str]:
    counts: dict[str, int] = {}
    for t in texts:
        for tok in _tokens(t):
            counts[tok] = counts.get(tok, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:k]]


def _video_title(messages: Sequence[Message]) -> str:
    m = re.search(r"^Video: (.*)$", _user(messages), re.M)
    return m.group(1).strip() if m else "This episode"


def metadata(messages: Sequence[Message]) -> str:
    rows = _plain(messages)
    texts = [t for _, t in rows]
    keys = _keywords(texts)
    scored = sorted(rows, key=lambda r: -_sentence_score(r[1]))
    quotes = [i for i, t in scored if 8 <= len(t.split()) <= 30][:3]
    takeaways = [t for _, t in scored if 6 <= len(t.split()) <= 28][:4]
    title = _video_title(messages)
    return json.dumps(
        {
            "title": title[:70],
            "description": f"{title}. In this video: {', '.join(keys[:5])}.",
            "tags": keys[:10],
            "summary": " ".join(texts[:3])[:500],
            "takeaways": takeaways,
            "quotes": quotes,
        }
    )


def blog_post(messages: Sequence[Message]) -> str:
    rows = _plain(messages)
    title = _video_title(messages)
    texts = [t for _, t in rows]
    out = [f"# {title}", "", " ".join(texts[:2]), ""]
    size = max(1, len(texts) // 4)
    for part in range(4):
        chunk = texts[part * size : (part + 1) * size]
        if not chunk:
            continue
        heading = " ".join(w.capitalize() for w in _keywords(chunk, 3)) or f"Part {part + 1}"
        out += [f"## {heading}", "", " ".join(chunk[:6]), ""]
    return "\n".join(out)


def social_posts(messages: Sequence[Message]) -> str:
    rows = _plain(messages)
    best = max((t for _, t in rows), key=_sentence_score, default="")
    title = _video_title(messages)
    tags = " ".join("#" + w for w in _keywords([t for _, t in rows], 3))
    keys = re.findall(r'"(\w+)": ""', _system(messages)) or [
        "x",
        "linkedin",
        "instagram",
        "tiktok",
        "threads",
        "facebook",
    ]
    out = {}
    for key in keys:
        if key == "x":
            out[key] = f"{best[:200]} {tags}".strip()
        elif key == "linkedin":
            out[key] = f"New episode: {title}.\n\n{best}\n\n{tags}"
        else:
            out[key] = f"{best} {tags}".strip()
    return json.dumps(out)


def judge_clips(messages: Sequence[Message]) -> str:
    """Offline judge: the deterministic lint decides "standalone"."""
    items = re.findall(r"^### Clip (\S+)\n(.*?)(?=^### Clip |\Z)", _user(messages), re.S | re.M)
    verdicts = []
    for cid, body in items:
        text = body.strip()
        token = _first_token(text)
        standalone = not (token in CONNECTORS or token in PRONOUN_STARTS or CONTEXT_PHRASES.search(text))
        ends = text.rstrip().endswith((".", "!"))
        verdicts.append(
            {
                "id": cid,
                "standalone": standalone,
                "complete_ending": ends,
                "hook": 6 if HOOK_WORDS.search(text[:160]) else 4,
                "payoff": 6 if ends else 3,
                "reason": "rules",
            }
        )
    return json.dumps({"clips": verdicts})


HANDLERS: dict[str, FakeHandler] = {
    "suggest_clips": suggest_clips,
    "chapters": chapters,
    "metadata": metadata,
    "blog_post": blog_post,
    "social_posts": social_posts,
    "judge_clips": judge_clips,
}
