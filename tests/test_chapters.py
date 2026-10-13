from __future__ import annotations

import itertools
import json

from recut.chapters import format_youtube, make_chapters, target_count, texttiling, validate, youtube_ready
from recut.config import Settings
from recut.models import Chapter
from recut.text import sentences_from_words
from tests.conftest import make_words
from tests.stubs import ScriptedModel

TOPICS = [
    "Espresso beans need a roast date. Fresh beans taste sweet. Old beans taste flat and bitter. Buy beans roasted this month. Store beans in a sealed jar.",
    "Grind size controls the shot time. A finer grind slows the water. A coarser grind speeds it up. Change the grind one step at a time. Aim for thirty seconds.",
    "Tamping must be level and firm. A tilted tamp sends water down one side. Use a tamper that fits the basket. Press with the same force every time. Check the puck after.",
    "Cleaning the machine weekly matters. Old coffee oil turns rancid. Backflush the group head with detergent. Wipe the steam wand after every use. Descale every few months.",
]


def _sentences():  # type: ignore[no-untyped-def]
    words = []
    t = 0.0
    for _round in range(3):
        for topic in TOPICS:
            words += make_words(topic, start=t, word=1.5)
            t = words[-1].end + 1.0
    return sentences_from_words(words)


def test_validate_forces_zero_start_merges_short_and_caps_titles() -> None:
    sentences = _sentences()
    raw = [
        (3, "1. Fresh beans"),
        (5, "Too close"),
        (20, "Chapter 2: Grinding " + "x" * 80),
        (21, "dupe-ish"),
        (9999, "bad index"),
    ]
    chapters = validate(raw, sentences, duration=sentences[-1].end)
    assert chapters[0].start == 0 and chapters[0].title == "Fresh beans"
    assert all(b.start - a.start >= 45 for a, b in itertools.pairwise(chapters))
    assert all(len(c.title) <= 60 for c in chapters)
    assert not any(c.title.startswith("Chapter") for c in chapters)


def test_validate_adds_an_introduction_when_the_first_chapter_is_late() -> None:
    sentences = _sentences()
    late = next(s.index for s in sentences if s.start > 100)
    chapters = validate([(late, "Grinding")], sentences, duration=sentences[-1].end)
    assert chapters[0].title == "Introduction" and chapters[0].start == 0


def test_youtube_rules() -> None:
    good = [Chapter(start=0, title="A"), Chapter(start=60, title="B"), Chapter(start=120, title="C")]
    assert youtube_ready(good, 200) == []
    assert any("three" in p for p in youtube_ready(good[:2], 200))
    assert any("00:00" in p for p in youtube_ready([Chapter(start=5, title="A"), *good[1:]], 200))
    assert any("10 seconds" in p for p in youtube_ready([*good, Chapter(start=195, title="D")], 200))
    assert format_youtube(good) == "00:00 A\n01:00 B\n02:00 C"


def test_target_count_bounds() -> None:
    assert target_count(60) == 3
    assert target_count(3600) == 15
    assert target_count(1200) == 10


def test_texttiling_finds_topic_shifts() -> None:
    sentences = _sentences()
    tiles = texttiling(sentences, k=6)
    assert tiles[0][0] == 0
    starts = [sentences[i].start for i, _ in tiles]
    assert len(starts) >= 3
    assert all(isinstance(t, str) and t for _, t in tiles)


def test_make_chapters_uses_the_model_and_validates(settings: Settings) -> None:
    sentences = _sentences()
    reply = {
        "chapters": [
            {"start": 0, "title": "Beans"},
            {"start": 5, "title": "Grind"},
            {"start": 10, "title": "Tamping"},
            {"start": 15, "title": "Cleaning"},
        ]
    }
    model = ScriptedModel({"chapters": json.dumps(reply)})
    chapters, stats = make_chapters(sentences, model, settings, duration=sentences[-1].end)
    assert [c.title for c in chapters][:2] == ["Beans", "Grind"]
    assert stats["calls"] == 1


def test_make_chapters_falls_back_to_texttiling_when_the_model_fails(settings: Settings) -> None:
    sentences = _sentences()
    chapters, stats = make_chapters(sentences, ScriptedModel(fail=True), settings, duration=sentences[-1].end)
    assert stats["fallback"] == "texttiling" and chapters and chapters[0].start == 0


def test_make_chapters_fills_up_when_the_model_returns_too_few(settings: Settings) -> None:
    sentences = _sentences()
    model = ScriptedModel({"chapters": json.dumps({"chapters": [{"start": 0, "title": "Everything"}]})})
    chapters, _ = make_chapters(sentences, model, settings, duration=sentences[-1].end)
    assert len(chapters) >= 3
