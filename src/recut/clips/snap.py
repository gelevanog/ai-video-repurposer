"""Cut points: sentence ranges to clip boundaries that never fall inside a word.

The model picks sentences; code decides the exact cut. A clip starts in the silence before its first word and ends in
the silence after its last word, at the quietest 30 ms of that gap (from the audio level, not only from Whisper's
word times, which are off by up to ~0.2 s). Length limits are enforced by adding or dropping whole sentences."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, field

from recut.media.levels import Levels
from recut.models import CutCheck, Sentence, Word

MIN_GAP = 0.03
"""A cut closer than this to a word edge counts as touching the word."""
SEARCH_IN = 0.45
SEARCH_OUT = 0.70


@dataclass
class SnapResult:
    first_sentence: int
    last_sentence: int
    start: float
    end: float
    cut_in: CutCheck
    cut_out: CutCheck
    adjustments: list[str] = field(default_factory=list)


def fit_length(
    first: int, last: int, sentences: Sequence[Sentence], *, min_s: float, max_s: float
) -> tuple[int, int, list[str]]:
    """Grow or shrink a sentence range to [min_s, max_s] by whole sentences. Keeps the first sentence (the hook)."""
    notes: list[str] = []
    n = len(sentences)
    first = max(0, min(first, n - 1))
    last = max(first, min(last, n - 1))

    def dur(a: int, b: int) -> float:
        return sentences[b].end - sentences[a].start

    original = dur(first, last)
    while dur(first, last) < min_s:
        grow_after = last + 1 < n and dur(first, last + 1) <= max_s
        grow_before = first > 0 and dur(first - 1, last) <= max_s
        if grow_after:
            last += 1
        elif grow_before:
            first -= 1
        else:
            break
    while dur(first, last) > max_s and last > first:
        if dur(first, last - 1) >= min_s or last - 1 == first:
            last -= 1
        elif first + 1 <= last and dur(first + 1, last) >= min_s:
            first += 1
        else:
            last -= 1
    final = dur(first, last)
    if abs(final - original) > 0.05:
        verb = "extended" if final > original else "shortened"
        notes.append(f"{verb} from {original:.1f} s to {final:.1f} s by whole sentences")
    return first, last, notes


def _word_bounds(words: Sequence[Word]) -> tuple[list[float], list[float]]:
    return [w.start for w in words], [w.end for w in words]


def check_cut(t: float, words: Sequence[Word], levels: Levels | None, *, speech_db: float | None = None) -> CutCheck:
    starts, ends = _word_bounds(words)
    i = bisect.bisect_right(starts, t) - 1  # last word starting at or before t
    inside = i >= 0 and words[i].start < t < words[i].end
    prev_end = ends[i] if i >= 0 else 0.0
    if inside:
        gap_before = -(t - words[i].start)
        gap_after = -(words[i].end - t)
    else:
        gap_before = t - prev_end if i >= 0 else t
        gap_after = (starts[i + 1] - t) if i + 1 < len(words) else 999.0
    level = None
    if levels is not None and len(levels.db):
        reference = speech_db if speech_db is not None else levels.speech_level()
        level = round(levels.at(t) - reference, 1)
    return CutCheck(
        at=round(t, 3),
        inside_word=inside,
        gap_before=round(gap_before, 3),
        gap_after=round(gap_after, 3),
        level_db=level,
    )


def cut_in_point(first_word: int, words: Sequence[Word], levels: Levels | None, pad: float) -> float:
    w = words[first_word]
    prev_end = words[first_word - 1].end if first_word > 0 else 0.0
    gap = w.start - prev_end
    lo = prev_end + min(MIN_GAP, gap / 2)
    hi = w.start - min(MIN_GAP, gap / 2)
    target = max(lo, w.start - pad)
    if levels is not None and len(levels.db) and hi > lo:
        lo = max(lo, w.start - SEARCH_IN)
        if hi > lo:
            return levels.quietest(lo, hi, prefer=target)
    return round(min(max(target, lo), hi) if hi >= lo else (prev_end + w.start) / 2, 3)


def cut_out_point(
    last_word: int, words: Sequence[Word], levels: Levels | None, pad: float, media_end: float | None
) -> float:
    w = words[last_word]
    next_start = words[last_word + 1].start if last_word + 1 < len(words) else (media_end or w.end + pad + 0.5)
    gap = next_start - w.end
    lo = w.end + min(MIN_GAP, gap / 2)
    hi = next_start - min(MIN_GAP, gap / 2)
    target = min(hi, w.end + pad)
    if levels is not None and len(levels.db) and hi > lo:
        hi = min(hi, w.end + SEARCH_OUT)
        if hi > lo:
            return levels.quietest(lo, hi, prefer=target)
    return round(max(min(target, hi), lo) if hi >= lo else (w.end + next_start) / 2, 3)


def snap(
    first: int,
    last: int,
    sentences: Sequence[Sentence],
    words: Sequence[Word],
    levels: Levels | None,
    *,
    min_s: float,
    max_s: float,
    pad_in: float,
    pad_out: float,
    media_end: float | None = None,
) -> SnapResult:
    first, last, notes = fit_length(first, last, sentences, min_s=min_s, max_s=max_s)
    fw, lw = sentences[first].first_word, sentences[last].last_word
    start = cut_in_point(fw, words, levels, pad_in)
    end = cut_out_point(lw, words, levels, pad_out, media_end)
    speech_db = levels.speech_level() if levels is not None and len(levels.db) else None
    return SnapResult(
        first,
        last,
        round(start, 3),
        round(end, 3),
        check_cut(start, words, levels, speech_db=speech_db),
        check_cut(end, words, levels, speech_db=speech_db),
        notes,
    )


def snap_times(
    start: float,
    end: float,
    sentences: Sequence[Sentence],
    words: Sequence[Word],
    levels: Levels | None,
    *,
    min_s: float,
    max_s: float,
    pad_in: float,
    pad_out: float,
    to_sentences: bool = True,
    media_end: float | None = None,
) -> SnapResult:
    """Snap free-form times (from the editor or an MCP client). With to_sentences the range widens to whole
    sentences; without, it snaps to the nearest word gap (an editor trim inside a sentence is allowed but flagged)."""
    if not sentences:
        raise ValueError("no sentences to snap to")
    if to_sentences:
        first = _sentence_at(start, sentences, prefer="start")
        last = max(first, _sentence_at(end, sentences, prefer="end"))
        return snap(
            first,
            last,
            sentences,
            words,
            levels,
            min_s=min_s,
            max_s=max_s,
            pad_in=pad_in,
            pad_out=pad_out,
            media_end=media_end,
        )
    fw = _word_at(start, words, prefer="start")
    lw = max(fw, _word_at(end, words, prefer="end"))
    s_in = cut_in_point(fw, words, levels, pad_in)
    s_out = cut_out_point(lw, words, levels, pad_out, media_end)
    first = _sentence_of_word(fw, sentences)
    last = _sentence_of_word(lw, sentences)
    speech_db = levels.speech_level() if levels is not None and len(levels.db) else None
    notes = []
    if sentences[first].first_word != fw:
        notes.append("starts inside a sentence")
    if sentences[last].last_word != lw:
        notes.append("ends inside a sentence")
    return SnapResult(
        first,
        last,
        s_in,
        s_out,
        check_cut(s_in, words, levels, speech_db=speech_db),
        check_cut(s_out, words, levels, speech_db=speech_db),
        notes,
    )


def _sentence_at(t: float, sentences: Sequence[Sentence], *, prefer: str) -> int:
    for s in sentences:
        if s.start <= t <= s.end:
            return s.index
    # in a gap: the next sentence for a start, the previous one for an end
    if prefer == "start":
        for s in sentences:
            if s.start >= t:
                return s.index
        return sentences[-1].index
    for s in reversed(sentences):
        if s.end <= t:
            return s.index
    return sentences[0].index


def _word_at(t: float, words: Sequence[Word], *, prefer: str) -> int:
    """The word containing t; in a gap the next word (for a start) or the previous one (for an end). A word that
    is cut through is kept whole: a start inside a word includes it, an end inside a word includes it."""
    starts = [w.start for w in words]
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and words[i].start <= t <= words[i].end:
        return i
    if prefer == "start":
        return min(i + 1, len(words) - 1)
    return max(i, 0)


def _sentence_of_word(wi: int, sentences: Sequence[Sentence]) -> int:
    for s in sentences:
        if s.first_word <= wi <= s.last_word:
            return s.index
    return sentences[-1].index
