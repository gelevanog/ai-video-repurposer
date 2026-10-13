from __future__ import annotations

import re

import pytest

from recut.render.captions import STYLES, TimedWord, build_ass, clip_words, escape, group_words, wrap_title
from tests.conftest import make_words

WORDS = make_words(
    "Here is the uncomfortable truth. Most creators price from fear, not from value. Nobody pays you for hours!"
)


def _timed() -> list[TimedWord]:
    return clip_words(WORDS, 0.0, WORDS[-1].end + 0.5)


def test_clip_words_are_relative_and_overrides_apply() -> None:
    start = WORDS[3].start
    words = clip_words(WORDS, start, WORDS[8].end, {4: "TRUTH", 5: ""})
    assert words[0].text == "uncomfortable" and words[0].start == 0.0
    assert words[1].text == "TRUTH"
    assert "Most" not in [w.text for w in words]  # deleted in the editor


def test_groups_respect_word_and_character_limits_and_punctuation() -> None:
    spec = STYLES["bold"]
    groups = group_words(_timed(), spec)
    assert all(len(g) <= spec.max_words for g in groups)
    assert all(len(" ".join(w.text for w in g)) <= spec.max_chars or len(g) == 1 for g in groups)
    for g in groups[:-1]:
        if g[-1].text.endswith("."):
            continue
    assert any(g[-1].text == "truth." for g in groups)  # sentence end closes a group


def test_pop_style_shows_one_word_at_a_time() -> None:
    assert all(len(g) == 1 for g in group_words(_timed(), STYLES["pop"]))


@pytest.mark.parametrize("style", list(STYLES))
def test_every_style_builds_a_valid_script(style: str) -> None:
    ass = build_ass(_timed(), style=style, duration=20.0, layout="track", title="Why your price is too low")  # type: ignore[arg-type]
    assert "[V4+ Styles]" in ass and "PlayResY: 1920" in ass
    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(events) >= 4
    assert sum(1 for e in events if ",Title," in e) == 1
    for event in events:
        start, end = event.split(",")[1:3]
        assert start < end or len(start) != len(end)


def test_highlight_follows_the_spoken_word() -> None:
    ass = build_ass(_timed(), style="bold", duration=20.0, layout="fit")
    events = [line for line in ass.splitlines() if line.startswith("Dialogue: 0")]
    first = events[0]
    assert r"\1c&H0000E1FF" in first and "HERE" in first
    assert "UNCOMFORTABLE" in ass and "truth." not in ass  # caps, punctuation stripped


def test_karaoke_uses_fill_timing() -> None:
    ass = build_ass(_timed(), style="karaoke", duration=20.0, layout="fit")
    assert re.search(r"\\kf\d+", ass)


def test_minimal_style_keeps_case_and_punctuation() -> None:
    ass = build_ass(_timed(), style="minimal", duration=20.0, layout="fit")
    assert "truth." in ass and "UNCOMFORTABLE" not in ass


def test_no_title_card_when_empty() -> None:
    ass = build_ass(_timed(), style="bold", duration=20.0, layout="fit", title="  ")
    assert ",Title," not in ass.split("[Events]")[1]


def test_text_is_escaped_for_ass() -> None:
    assert escape("{\\an8}hack") == "(/an8)hack"
    ass = build_ass([TimedWord(0, 1, "{bad}")], style="minimal", duration=2.0, layout="fit")
    assert "{bad}" not in ass


def test_title_wrapping() -> None:
    wrapped = wrap_title(
        "The three number pricing test that changed how I quote every single project forever", width=20, max_lines=2
    )
    assert wrapped.count("\\N") == 1 and wrapped.endswith("…")
