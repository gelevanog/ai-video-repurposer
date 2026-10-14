from __future__ import annotations

import json

from recut.clips.suggest import (
    _grounded_hook,
    build_clips,
    clips_wanted,
    parse_candidates,
    parse_lines,
    rank_score,
    suggest,
    transcript_lines,
    windows,
)
from recut.config import Settings
from recut.fake_tasks import HANDLERS
from recut.llm.fake import FakeModel
from recut.models import Scores, Transcript
from recut.text import sentences_from_words
from tests.conftest import make_words, silent_levels_for
from tests.stubs import ScriptedModel


def test_transcript_lines_round_trip(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    _, sentences = talk
    text = transcript_lines(sentences)
    parsed = parse_lines(text)
    assert [p[0] for p in parsed] == [s.index for s in sentences]
    assert abs(parsed[3][1] - sentences[3].start) < 0.11
    assert parsed[0][4] == sentences[0].text


def test_parse_candidates_validates_ranges_and_scores() -> None:
    reply = {
        "clips": [
            {
                "start": 2,
                "end": 5,
                "title": "Good",
                "scores": {"hook": 14, "payoff": "7", "standalone": None},
                "virality": 250,
            },
            {"start": 9, "end": 4, "title": "Reversed"},
            {"start": 40, "end": 41, "title": "Out of range"},
            {"start": "x", "end": 2},
            "not an object",
        ]
    }
    found, rejected = parse_candidates(reply, set(range(12)))
    assert [(c.first, c.last) for c in found] == [(2, 5), (4, 9)]
    assert found[0].scores.hook == 10 and found[0].scores.payoff == 7 and found[0].scores.virality == 100
    assert found[0].scores.standalone == 0
    assert len(rejected) == 3


def test_parse_candidates_without_a_list() -> None:
    found, rejected = parse_candidates({"answer": "none"}, {1})
    assert found == [] and rejected


def test_rank_score_rewards_scores_and_penalizes_issues() -> None:
    great = Scores(hook=9, payoff=9, standalone=9, clarity=9, virality=90)
    assert rank_score(great, 0) > rank_score(great, 30)
    assert rank_score(Scores(), 0) == 0
    assert rank_score(Scores(hook=10, payoff=10, standalone=10, clarity=10, virality=100), 0) == 100


def test_hook_must_be_the_actual_opening() -> None:
    opening = "Here is the truth about pricing that nobody tells you."
    assert _grounded_hook("the truth about pricing", opening) == "the truth about pricing"
    assert _grounded_hook("You will not believe this!", opening) == opening


def test_windows_split_long_transcripts_with_overlap() -> None:
    words = []
    t = 0.0
    for i in range(400):
        words += make_words(f"Sentence number {i} is here.", start=t)
        t = words[-1].end + 1.5
    sentences = sentences_from_words(words)
    parts = windows(sentences, minutes=5)
    assert len(parts) > 3
    assert parts[1][0].start < parts[0][-1].end  # overlap
    assert {s.index for p in parts for s in p} == {s.index for s in sentences}
    assert len(windows(sentences[:20], minutes=5)) == 1


def test_clips_wanted_scales_with_duration() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert clips_wanted(60, s) == 2
    assert clips_wanted(3600, s) == s.max_clips
    assert clips_wanted(1200, s) == 6


def test_build_clips_snaps_lints_and_drops_overlaps(talk: tuple[Transcript, list], settings: Settings) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    reply = {
        "clips": [
            {
                "start": 0,
                "end": 4,
                "title": "Price the result",
                "hook": "Here is the truth",
                "scores": {"hook": 9, "payoff": 9, "standalone": 9, "clarity": 8},
                "virality": 80,
            },
            {
                "start": 1,
                "end": 4,
                "title": "Duplicate",
                "scores": {"hook": 3, "payoff": 3, "standalone": 3, "clarity": 3},
                "virality": 10,
            },
            {
                "start": 10,
                "end": 14,
                "title": "Three numbers",
                "scores": {"hook": 8, "payoff": 8, "standalone": 8, "clarity": 8},
                "virality": 70,
            },
        ]
    }
    found, _ = parse_candidates(reply, {s.index for s in sentences})
    clips, rejected = build_clips(
        found, sentences, transcript.words, silent_levels_for(transcript.words), settings, source="model"
    )
    assert sorted(c.title for c in clips) == ["Price the result", "Three numbers"]
    assert any("overlaps" in r for r in rejected)
    assert [c.rank for c in clips] == [1, 2]
    for clip in clips:
        assert 20 <= clip.duration <= 60
        assert clip.cut_in and not clip.cut_in.inside_word
        assert clip.title_card_text == clip.title


def test_suggest_with_the_offline_model(talk: tuple[Transcript, list], settings: Settings) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    report = suggest(sentences, transcript.words, None, FakeModel(HANDLERS), settings, title="Pricing")
    assert report.clips and report.calls == 1
    assert all(c.source == "rules" for c in report.clips)
    assert report.clips[0].first_sentence in {0, 2, 7, 10}


def test_suggest_survives_a_failing_model(talk: tuple[Transcript, list], settings: Settings) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    model = ScriptedModel(fail=True)
    report = suggest(sentences, transcript.words, None, model, settings)
    assert report.clips == [] and "boom" in report.rejected[0]
    assert len(model.calls) == 2  # one retry asking for a direct answer
    assert "answer with the JSON directly" in model.calls[1][1][1]["content"]


def test_suggest_prompt_contains_numbered_sentences(talk: tuple[Transcript, list], settings: Settings) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    model = ScriptedModel({"suggest_clips": json.dumps({"clips": [{"start": 0, "end": 3, "title": "T"}]})})
    report = suggest(sentences, transcript.words, None, model, settings, title="My video")
    task, messages = model.calls[0]
    assert task == "suggest_clips"
    assert "[0] 00:00.0" in messages[1]["content"] and "Video: My video" in messages[1]["content"]
    assert report.clips[0].source == "model"


def test_offline_suggester_avoids_connector_openings(talk: tuple[Transcript, list], settings: Settings) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    report = suggest(sentences, transcript.words, None, FakeModel(HANDLERS), settings)
    for clip in report.clips:
        assert not sentences[clip.first_sentence].text.startswith(("And", "As I said", "But"))
