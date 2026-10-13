"""Burned-in, word-level animated captions as an ASS subtitle script (rendered by libass inside ffmpeg).

Words are grouped into short lines (2-4 words, broken at punctuation and pauses), and the word being spoken is
highlighted with a colour change and a short scale "pop". Five styles: bold, boxed, karaoke, pop and minimal. The
fonts (Poppins ExtraBold, Archivo Black, Anton; SIL Open Font License) ship in assets/fonts so the output looks the
same on every machine."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from recut.models import CaptionStyle, Word


@dataclass(frozen=True)
class StyleSpec:
    font: str
    size: int
    primary: str
    """ASS colour &HAABBGGRR of the text."""
    highlight: str
    outline_colour: str
    back_colour: str
    border_style: int
    outline: float
    shadow: float
    max_words: int
    max_chars: int
    uppercase: bool
    strip_punct: bool
    pop: bool
    karaoke: bool = False
    one_word: bool = False
    lines: int = 1


STYLES: dict[str, StyleSpec] = {
    "bold": StyleSpec(
        "Poppins ExtraBold", 84, "&H00FFFFFF", "&H0000E1FF", "&H00000000", "&H96000000", 1, 6, 2, 3, 18, True, True, True
    ),
    "boxed": StyleSpec(
        "Archivo Black", 68, "&H00FFFFFF", "&H0066F2A6", "&H50201812", "&H00000000", 3, 14, 0, 4, 22, False, False, True
    ),
    "karaoke": StyleSpec(
        "Poppins ExtraBold",
        74,
        "&H00FFE14D",
        "&H00FFFFFF",
        "&H00141414",
        "&H80000000",
        1,
        5,
        1,
        6,
        30,
        False,
        False,
        False,
        karaoke=True,
        lines=2,
    ),
    "pop": StyleSpec(
        "Anton", 132, "&H00FFFFFF", "&H0000E1FF", "&H00000000", "&H96000000", 1, 8, 3, 1, 14, True, True, True, one_word=True
    ),
    "minimal": StyleSpec(
        "Poppins ExtraBold", 54, "&H00FFFFFF", "&H00FFFFFF", "&H64000000", "&H78000000", 1, 2.5, 1.5, 7, 34, False, False, False, lines=2
    ),
}
STYLE_LABELS: dict[str, str] = {
    "bold": "Bold: white caps, yellow active word",
    "boxed": "Boxed: text on a dark box, green active word",
    "karaoke": "Karaoke: two lines, the colour sweeps with the voice",
    "pop": "Pop: one big word at a time",
    "minimal": "Minimal: small sentence-case subtitles",
}

PLAY_W, PLAY_H = 1080, 1920
_PUNCT_END = re.compile(r"[.!?…]$")
_SOFT_BREAK = re.compile(r"[,;:—–-]$")


@dataclass
class TimedWord:
    start: float
    end: float
    text: str


def clip_words(
    words: Sequence[Word], start: float, end: float, overrides: dict[int, str] | None = None
) -> list[TimedWord]:
    """Words inside [start, end], times relative to the clip start, caption-editor corrections applied."""
    out: list[TimedWord] = []
    overrides = overrides or {}
    for i, w in enumerate(words):
        if w.end <= start or w.start >= end:
            continue
        text = overrides.get(i, w.text).strip()
        if not text:
            continue  # a word deleted in the caption editor
        out.append(TimedWord(max(0.0, w.start - start), min(end, w.end) - start, text))
    return out


def group_words(words: Sequence[TimedWord], spec: StyleSpec) -> list[list[TimedWord]]:
    if spec.one_word:
        return [[w] for w in words]
    groups: list[list[TimedWord]] = []
    current: list[TimedWord] = []
    chars = 0
    max_words = spec.max_words * spec.lines if spec.lines > 1 else spec.max_words
    max_chars = spec.max_chars * spec.lines if spec.lines > 1 else spec.max_chars
    for i, w in enumerate(words):
        if current:
            pause = w.start - current[-1].end
            prev = current[-1].text
            if (
                len(current) >= max_words
                or chars + 1 + len(w.text) > max_chars
                or _PUNCT_END.search(prev)
                or pause > 0.6
                or (_SOFT_BREAK.search(prev) and len(current) >= 2)
            ):
                groups.append(current)
                current, chars = [], 0
        current.append(w)
        chars += len(w.text) + (1 if chars else 0)
        if i == len(words) - 1:
            groups.append(current)
    return groups


def _display(text: str, spec: StyleSpec) -> str:
    if spec.strip_punct:
        text = re.sub(r"^[\"'“‘(]+|[\"'”’),.;:!?…]+$", "", text) or text
    if spec.uppercase:
        text = text.upper()
    return escape(text)


def escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def _t(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _line_break(tokens: list[str], spec: StyleSpec) -> list[str]:
    """For two-line styles: a \\N between the halves of a long group."""
    if spec.lines < 2 or len(tokens) <= spec.max_words:
        return tokens
    half = (len(tokens) + 1) // 2
    return [*tokens[: half - 1], tokens[half - 1] + "\\N", *tokens[half:]]


def wrap_title(text: str, width: int = 22, max_lines: int = 3) -> str:
    words = text.split()
    lines: list[str] = []
    line = ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;:") + "…"
    return "\\N".join(escape(x) for x in lines)


def build_ass(
    words: Sequence[TimedWord],
    *,
    style: CaptionStyle,
    duration: float,
    layout: str,
    title: str = "",
    title_seconds: float = 3.0,
) -> str:
    spec = STYLES[style]
    # caption baseline: lower third over a tracked face; under the picture for the full-frame "fit" layout
    margin_v = 470 if layout == "fit" else 560
    title_margin = 300 if layout == "fit" else 250
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_W}
PlayResY: {PLAY_H}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{spec.font},{spec.size},{spec.primary},{spec.highlight},{spec.outline_colour},{spec.back_colour},0,0,0,0,100,100,0,0,{spec.border_style},{spec.outline},{spec.shadow},2,70,70,{margin_v},1
Style: Title,Poppins ExtraBold,62,&H00151515,&H00151515,&H00FFFFFF,&H64000000,0,0,0,0,100,100,0,0,3,22,0,8,90,90,{title_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    if title.strip():
        end = min(title_seconds, max(1.0, duration - 0.2))
        events.append(f"Dialogue: 1,{_t(0)},{_t(end)},Title,,0,0,0,,{{\\fad(180,260)}}{wrap_title(title)}")
    groups = group_words(words, spec)
    for gi, group in enumerate(groups):
        g_start = group[0].start
        nxt = groups[gi + 1][0].start if gi + 1 < len(groups) else None
        g_end = group[-1].end + 0.25
        if nxt is not None and nxt - group[-1].end < 0.5:
            g_end = nxt
        if nxt is not None:
            g_end = min(g_end, nxt)
        g_end = min(g_end, duration)
        if g_end <= g_start:
            continue
        tokens = [_display(w.text, spec) for w in group]
        if spec.karaoke:
            parts = []
            for wi, w in enumerate(group):
                upto = group[wi + 1].start if wi + 1 < len(group) else w.end
                k = max(1, round((upto - w.start) * 100))
                parts.append(f"{{\\kf{k}}}{tokens[wi]}")
            lead = max(0, round((group[0].start - g_start) * 100))
            text = (f"{{\\k{lead}}}" if lead else "") + " ".join(_line_break(parts, spec))
            events.append(f"Dialogue: 0,{_t(g_start)},{_t(g_end)},Cap,,0,0,0,,{{\\fad(60,60)}}{text}")
            continue
        if spec.one_word:
            w = group[0]
            colour = spec.highlight if _emphasis(w.text) else spec.primary
            anim = "{\\fscx82\\fscy82\\t(0,90,\\fscx104\\fscy104)\\t(90,150,\\fscx100\\fscy100)" + f"\\1c{colour}" + "}"
            events.append(f"Dialogue: 0,{_t(g_start)},{_t(g_end)},Cap,,0,0,0,,{anim}{tokens[0]}")
            continue
        if not spec.pop:
            text = " ".join(_line_break(tokens, spec))
            events.append(f"Dialogue: 0,{_t(g_start)},{_t(g_end)},Cap,,0,0,0,,{text}")
            continue
        for wi, w in enumerate(group):
            e_start = g_start if wi == 0 else w.start
            e_end = group[wi + 1].start if wi + 1 < len(group) else g_end
            if e_end <= e_start:
                continue
            styled = []
            for ti, tok in enumerate(tokens):
                if ti == wi:
                    styled.append(
                        f"{{\\1c{spec.highlight}\\fscx100\\fscy100\\t(0,70,\\fscx112\\fscy112)\\t(70,140,\\fscx106\\fscy106)}}"
                        f"{tok}{{\\1c{spec.primary}\\fscx100\\fscy100}}"
                    )
                else:
                    styled.append(tok)
            events.append(f"Dialogue: 0,{_t(e_start)},{_t(e_end)},Cap,,0,0,0,,{' '.join(styled)}")
    return header + "\n".join(events) + "\n"


def _emphasis(text: str) -> bool:
    core = re.sub(r"\W", "", text)
    return bool(re.search(r"\d", core)) or len(core) >= 9
