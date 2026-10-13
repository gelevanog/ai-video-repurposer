"""Deterministic checks that a clip can stand on its own: length, how it starts, how it ends, references to context
the viewer has not seen, cut quality. The model's own "standalone" score is not trusted alone; these are shown in
the review UI next to it and lower the ranking score."""

from __future__ import annotations

import re
from collections.abc import Sequence

from recut.models import ClipIssue, CutCheck, Sentence, Word

CONNECTORS = {"and", "but", "so", "because", "or", "also", "then", "plus", "which", "anyway", "besides"}
PRONOUN_STARTS = {"he", "she", "they", "it", "him", "her", "them", "his", "their", "its"}
DEMONSTRATIVE_STARTS = {"this", "that", "these", "those", "there"}
SELF_INTRODUCING = re.compile(
    r"^(this|that)(?:'s| is| was)? (?:is |was )?(?:the|a|an|my|our|why|how|what|where|when|exactly)\b", re.IGNORECASE
)
CONTEXT_PHRASES = re.compile(
    r"\b(as i (?:said|mentioned)|like i (?:said|mentioned)|i mentioned (?:earlier|before)|as we (?:saw|discussed)|"
    r"going back to|earlier (?:i|we)|the (?:second|third|other) (?:one|thing|point)|that (?:point|story) i)\b",
    re.IGNORECASE,
)
GREETINGS = re.compile(r"^(hi|hello|hey|welcome|good (?:morning|afternoon|evening)|thanks for (?:having|joining))\b", re.I)
FILLERS = {"um", "uh", "erm", "uhm", "hmm"}


def _first_token(text: str) -> str:
    match = re.match(r"\W*([A-Za-z']+)", text)
    return match.group(1).lower() if match else ""


def lint_clip(
    first: int,
    last: int,
    sentences: Sequence[Sentence],
    words: Sequence[Word],
    *,
    start: float,
    end: float,
    min_s: float,
    max_s: float,
    cut_in: CutCheck | None = None,
    cut_out: CutCheck | None = None,
) -> list[ClipIssue]:
    issues: list[ClipIssue] = []
    duration = end - start
    if duration < min_s - 0.5 or duration > max_s + 0.5:
        issues.append(
            ClipIssue(
                code="length",
                severity="error",
                message=f"{duration:.1f} s is outside {min_s:.0f}-{max_s:.0f} s",
            )
        )
    opening = sentences[first].text.strip()
    token = _first_token(opening)
    if token in CONNECTORS:
        issues.append(
            ClipIssue(
                code="starts_with_connector",
                severity="warning",
                message=f'starts with "{token}": may continue a thought from before the clip',
            )
        )
    elif token in PRONOUN_STARTS:
        issues.append(
            ClipIssue(
                code="dangling_reference",
                severity="warning",
                message=f'starts with "{token}": the viewer may not know who or what it refers to',
            )
        )
    elif token in DEMONSTRATIVE_STARTS and not SELF_INTRODUCING.match(opening):
        issues.append(
            ClipIssue(
                code="dangling_reference",
                severity="info",
                message=f'starts with "{token}": check that it does not point to something said before',
            )
        )
    if GREETINGS.match(opening):
        issues.append(ClipIssue(code="weak_hook", severity="warning", message="opens with a greeting, not a hook"))
    body = " ".join(s.text for s in sentences[first : last + 1])
    if (match := CONTEXT_PHRASES.search(body)) is not None:
        issues.append(
            ClipIssue(
                code="context_reference",
                severity="warning",
                message=f'"{match.group(0)}" refers to context outside the clip',
            )
        )
    closing = sentences[last].text.strip()
    if closing.endswith("?"):
        issues.append(
            ClipIssue(code="ends_on_question", severity="warning", message="ends on a question: the answer is cut")
        )
    elif not re.search(r"[.!…]['\")\]]*$", closing):
        issues.append(ClipIssue(code="ends_mid_sentence", severity="warning", message="ends without a full stop"))
    clip_words = words[sentences[first].first_word : sentences[last].last_word + 1]
    if clip_words:
        fillers = sum(1 for w in clip_words if re.sub(r"\W", "", w.text.lower()) in FILLERS)
        if fillers / len(clip_words) > 0.06:
            issues.append(
                ClipIssue(code="fillers", severity="info", message=f"{fillers} filler words ({fillers / len(clip_words):.0%})")
            )
        speakers = {w.speaker for w in clip_words}
        if len(speakers) > 2:
            issues.append(ClipIssue(code="many_speakers", severity="info", message=f"{len(speakers)} speakers"))
    for label, cut in (("in", cut_in), ("out", cut_out)):
        if cut is None:
            continue
        if cut.inside_word:
            issues.append(ClipIssue(code=f"cut_{label}_mid_word", severity="error", message=f"cut-{label} falls inside a word"))
        elif min(cut.gap_before, cut.gap_after) < 0.03:
            issues.append(
                ClipIssue(
                    code=f"cut_{label}_tight",
                    severity="warning",
                    message=f"cut-{label} is within 30 ms of a word (fast speech, no pause)",
                )
            )
        if cut.level_db is not None and cut.level_db > -12:
            issues.append(
                ClipIssue(
                    code=f"cut_{label}_loud",
                    severity="warning",
                    message=f"cut-{label} lands on audio {cut.level_db:+.0f} dB from speech level (likely speech or music)",
                )
            )
    return issues


PENALTY = {"error": 25.0, "warning": 6.0, "info": 1.0}


def penalty(issues: Sequence[ClipIssue]) -> float:
    return sum(PENALTY[i.severity] for i in issues)
