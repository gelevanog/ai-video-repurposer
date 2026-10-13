"""Grounding checks for generated text: every number must occur in the transcript, every quotation must be
something that was actually said. Findings are shown to the reviewer; nothing is silently rewritten."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from recut.models import GroundingIssue
from recut.text import normalize_for_wer

_NUMBER = re.compile(
    r"(?<![\w#@])(\d+(?:[.,]\d+)*)(\s?(?:k|K|m|M|bn)\b|\s(?:thousand|million|billion)\b)?(\s?%| percent)?"
)
_SCALE = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
}


def _value(raw: str, scale: str | None) -> str:
    """ "40", "k" -> "40000"; "1.5", "million" -> "1500000"; "4,000" -> "4000"."""
    number = raw.replace(",", "")
    if not scale:
        return number
    try:
        scaled = float(number) * _SCALE[scale.strip().lower()]
    except (KeyError, ValueError):
        return number
    return str(int(scaled)) if scaled.is_integer() else str(scaled)


_QUOTE = re.compile(r"[\"“]([^\"”]{12,400})[\"”]")
_LIST_COUNT = re.compile(
    r"^\s*(?:top\s+)?(\d+)\s+(?:key\s+|big\s+|simple\s+|practical\s+)?(?:takeaways?|lessons?|tips?|ways?|things|reasons|"
    r"steps|ideas|mistakes|questions|insights|points|highlights|moments)\b",
    re.I,
)
_TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
QUOTE_MIN_SCORE = 88.0


class TranscriptIndex:
    """The transcript in the normalized forms the checks compare against."""

    def __init__(self, text: str) -> None:
        self.tokens = normalize_for_wer(text)
        self.text = " ".join(self.tokens)
        self.numbers = {t for t in self.tokens if t.isdigit()}
        joined = re.sub(r"(\d),(\d{3})", r"\1\2", text)
        self.numbers |= {_value(m.group(1), m.group(2)) for m in _NUMBER.finditer(joined)}
        self.numbers |= {n.split(".")[0] for n in list(self.numbers)}


def check_numbers(text: str, index: TranscriptIndex) -> list[GroundingIssue]:
    issues: list[GroundingIssue] = []
    clean = _TIMESTAMP.sub(" ", text)
    seen: set[str] = set()
    for match in _NUMBER.finditer(clean):
        raw = match.group(1) + (match.group(2) or "")
        value = _value(match.group(1), match.group(2))
        if value in seen:
            continue
        seen.add(value)
        line_start = clean.rfind("\n", 0, match.start()) + 1
        if _LIST_COUNT.match(clean[line_start:]) or _LIST_COUNT.match(clean[match.start() :]):
            continue  # "3 takeaways": a count of the post's own list, not a claim from the video
        if value in index.numbers or (value.split(".")[0] in index.numbers and "." not in value):
            continue
        issues.append(
            GroundingIssue(
                kind="number",
                text=raw + (match.group(3) or ""),
                message=f'"{raw}" does not appear in the transcript: check it before publishing',
            )
        )
    return issues


def check_quotes(text: str, index: TranscriptIndex) -> list[GroundingIssue]:
    issues: list[GroundingIssue] = []
    for match in _QUOTE.finditer(text):
        quote = match.group(1).strip()
        norm = " ".join(normalize_for_wer(quote))
        if len(norm.split()) < 4:
            continue
        score = fuzz.partial_ratio(norm, index.text) if index.text else 0.0
        if score < QUOTE_MIN_SCORE:
            issues.append(
                GroundingIssue(
                    kind="quote",
                    text=quote[:120],
                    message=f"quoted text not found in the transcript (best match {score:.0f}%): paraphrase or fix it",
                )
            )
    return issues


def check(text: str, index: TranscriptIndex) -> list[GroundingIssue]:
    return check_numbers(text, index) + check_quotes(text, index)
