"""Sentences from timed words, timestamp formatting, and the text normalization used by WER and grounding checks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from rapidfuzz.distance import Levenshtein

from recut.models import Sentence, Word

_ABBREV = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
    "jr.",
    "sr.",
    "inc.",
    "no.",
    "approx.",
}
_END = re.compile(r"[.!?…]+[\"')\]]*$")
MAX_SENTENCE_WORDS = 45
LONG_PAUSE = 1.0


def is_sentence_end(token: str) -> bool:
    low = token.lower()
    if low in _ABBREV or re.fullmatch(r"(?:[a-z]\.){2,}", low):
        return False
    return bool(_END.search(token))


def sentences_from_words(words: Sequence[Word]) -> list[Sentence]:
    """Split at terminal punctuation; also at a speaker change, and at a long pause inside a very long run-on."""
    out: list[Sentence] = []
    first = 0
    for i, word in enumerate(words):
        last = i == len(words) - 1
        end = last or is_sentence_end(word.text)
        if not end:
            nxt = words[i + 1]
            if nxt.speaker != word.speaker or (i - first + 1 >= MAX_SENTENCE_WORDS and nxt.start - word.end >= 0.5) or (nxt.start - word.end >= LONG_PAUSE * 2.5 and i - first >= 3):
                end = True
        if end:
            span = words[first : i + 1]
            out.append(
                Sentence(
                    index=len(out),
                    start=span[0].start,
                    end=span[-1].end,
                    text=" ".join(w.text for w in span),
                    speaker=span[0].speaker,
                    first_word=first,
                    last_word=i,
                )
            )
            first = i + 1
    return out


def ts(seconds: float, *, millis: bool = False) -> str:
    """01:02:03 / 02:03 (YouTube style); with millis: 02:03.4."""
    seconds = max(0.0, seconds)
    whole = int(seconds)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    base = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    if millis:
        base += f".{int((seconds - whole) * 10)}"
    return base


_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
}
_FILLERS = {"um", "uh", "erm", "hmm", "mm", "ah", "eh", "uhm", "mhm"}
_CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
}


def normalize_for_wer(text: str) -> list[str]:
    """Lowercase, ASCII-fold, expand contractions, number words to digits, drop punctuation and fillers.

    Applied identically to the reference and the hypothesis, so formatting choices (\"10%\" vs \"ten percent\",
    \"don't\" vs \"do not\") do not count as recognition errors."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)  # [applause], (inaudible)
    text = text.replace("%", " percent ").replace("&", " and ").replace("’", "'")
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    for k in ("won't", "can't"):
        text = text.replace(k, _CONTRACTIONS[k])
    for suffix in ("n't", "'re", "'ve", "'ll", "'d", "'m"):
        text = text.replace(suffix, _CONTRACTIONS[suffix])
    text = re.sub(r"'s\b", "s", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for token in text.split():
        if token in _FILLERS:
            continue
        tokens.append(_NUMBER_WORDS.get(token, token))
    return tokens


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of the minimum alignment (Levenshtein over tokens, rapidfuzz)."""
    subs = dels = ins = 0
    for op in Levenshtein.editops(list(ref), list(hyp)):
        if op.tag == "replace":
            subs += 1
        elif op.tag == "delete":
            dels += 1
        elif op.tag == "insert":
            ins += 1
    return subs, dels, ins


def wer(reference: str, hypothesis: str) -> dict[str, float]:
    ref, hyp = normalize_for_wer(reference), normalize_for_wer(hypothesis)
    s, d, i = edit_distance(ref, hyp)
    return {
        "wer": (s + d + i) / max(1, len(ref)),
        "ref_words": float(len(ref)),
        "substitutions": float(s),
        "deletions": float(d),
        "insertions": float(i),
    }


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w[\w'’-]*\b", text))
