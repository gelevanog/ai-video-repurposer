from __future__ import annotations

import json
from pathlib import Path

import pytest

from recut.config import Settings
from recut.eval import report, testset
from recut.eval.judge import JudgeItem, judge
from recut.eval.runner import chapter_scores, highlight_recall, random_windows, speaker_accuracy
from recut.eval.scripts import validate
from recut.eval.synth import split_sentences
from recut.models import Clip, Project, SourceInfo, Transcript
from recut.text import sentences_from_words
from tests.conftest import LONG_TALK, make_words


def _clip(start: float, end: float) -> Clip:
    return Clip(id=f"c{start}", start=start, end=end, first_sentence=0, last_sentence=0, title="t")


def test_chapter_scores_with_tolerance() -> None:
    gold = [0, 100, 200, 300]
    scores = chapter_scores([0, 110, 260, 305, 400], gold, 500)
    assert scores["tol15"]["hits"] == 2 and scores["tol30"]["hits"] == 2
    assert scores["tol15"]["recall"] == round(2 / 3, 3) and scores["tol15"]["precision"] == 0.5
    assert scores["tol15"]["mean_abs_error_s"] == 7.5
    assert chapter_scores([0], [0], 10)["tol15"]["recall"] == 1.0


def test_highlight_recall() -> None:
    highlights = [{"start": 10, "end": 50}, {"start": 100, "end": 140}]
    out = highlight_recall([_clip(5, 40), _clip(130, 160)], highlights)
    assert out["found"] == 1 and out["recall"] == 0.5 and out["coverage"] == [0.75, 0.25]
    assert out["clips_mostly_on_a_highlight"] == 1


def test_speaker_accuracy_uses_the_best_label_mapping() -> None:
    words = make_words("one two. three four.", speaker="S2")
    for w in words[2:]:
        w.speaker = "S1"
    transcript = Transcript(duration=10, model="t", words=words, speakers=["S2", "S1"])
    project = Project(id="p-000001", name="p", source=SourceInfo(path="x", filename="x"), transcript=transcript)
    gold = {
        "speakers": {"A": "a", "B": "b"},
        "sentences": [
            {"start": words[0].start, "end": words[1].end, "speaker": "A"},
            {"start": words[2].start, "end": words[3].end, "speaker": "B"},
        ],
    }
    result = speaker_accuracy(project, gold)
    assert result["word_accuracy"] == 1.0 and result["count_correct"]


def test_random_windows_are_within_length(settings: Settings) -> None:
    words = make_words(LONG_TALK * 4)
    transcript = Transcript(duration=words[-1].end, model="t", words=words)
    project = Project(
        id="p-000002",
        name="p",
        source=SourceInfo(path="x", filename="x"),
        transcript=transcript,
        sentences=sentences_from_words(words),
    )
    windows = random_windows(project, 5, settings, seed=1)
    assert len(windows) == 5
    for i, j in windows:
        assert 20 <= project.sentences[j].end - project.sentences[i].start <= 60
    assert windows == random_windows(project, 5, settings, seed=1)


def test_judge_with_the_offline_rules(settings: Settings) -> None:
    items = [
        JudgeItem("m1", "model", "Here is the truth about pricing. Price the result, not the hours."),
        JudgeItem("r1", "random", "And as I said, he did that again"),
    ]
    verdicts = judge(items, settings, tag="test")
    assert verdicts["m1"]["standalone"] is True and verdicts["r1"]["standalone"] is False
    assert {v["group"] for v in verdicts.values()} == {"model", "random"}


def test_parse_srt_strips_markers() -> None:
    srt = "﻿1\n00:00:01,500 --> 00:00:03,236\n>> Behind me is\n\n2\n00:00:03,236 --> 00:00:05,338\n<i>a test</i>\n"
    cues = testset.parse_srt(srt)
    assert cues == [(1.5, 3.236, "Behind me is"), (3.236, 5.338, "a test")]


def test_manifest_lists_licensed_items_with_checksums() -> None:
    items = testset.load_manifest()
    assert len(items) >= 10
    for item in items:
        assert item["license"]
        if item["kind"] == "nasa":
            assert item["url"].startswith("https://images-assets.nasa.gov/") and len(item["sha256"]) == 64
            assert "NASA" in item["attribution"]
        else:
            assert (testset.REPO_ROOT / item["script"]).parent.exists()


def test_script_validation() -> None:
    outline = {"speakers": {"S1": {}, "S2": {}}, "chapters": [{"title": "a"}, {"title": "b"}]}
    turn = {"speaker": "S2", "text": "One. Two. Three. " + "word " * 80, "highlight": True}
    good = {"chapters": [{"turns": [turn, {"speaker": "S1", "text": "word " * 120}]} for _ in range(2)]}
    assert validate(good, outline) == []
    bad = {"chapters": [{"turns": [{"speaker": "S9", "text": "short"}]}]}
    problems = validate(bad, outline)
    assert any("unknown speaker" in p for p in problems) and any("highlights" in p for p in problems)


def test_split_sentences() -> None:
    assert split_sentences("I charged $200. It made $40,000! Really? Yes.") == [
        "I charged $200.",
        "It made $40,000!",
        "Really?",
        "Yes.",
    ]


def test_summary_folds_item_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    items = tmp_path / "demo" / "items"
    items.mkdir(parents=True)
    row = {
        "id": "a",
        "kind": "synthetic",
        "duration": 600.0,
        "asr_seconds": 30.0,
        "stages": {"clips": {"seconds": 10}, "chapters": {"seconds": 5}, "posts": {"seconds": 20}},
        "wer": {"wer": 0.05, "ref_words": 100.0, "substitutions": 3.0, "deletions": 1.0, "insertions": 1.0},
        "speakers": {
            "gold_speakers": 2,
            "predicted_speakers": 2,
            "count_correct": True,
            "word_accuracy": 0.9,
            "words": 100,
        },
        "chapters": {
            "predicted": 5,
            "gold": 5,
            "tol15": {"hits": 3, "mean_abs_error_s": 4.0},
            "tol30": {"hits": 4, "mean_abs_error_s": 6.0},
        },
        "highlights": {"highlights": 5, "found": 4, "clips_mostly_on_a_highlight": 3},
        "clips": [
            {
                "duration": 35.0,
                "issues": [],
                "cut_in": {"inside_word": False, "gap_before": 0.2, "gap_after": 0.1, "level_db": -40},
                "cut_out": {"inside_word": False, "gap_before": 0.3, "gap_after": 0.4, "level_db": -35},
                "render": {"seconds": 6.0, "layout": "fit"},
                "boundary": {"first_ok": True, "last_ok": False},
                "framing": None,
            }
        ],
        "judge": {
            "x": {"group": "model", "standalone": True, "complete_ending": True},
            "y": {"group": "random", "standalone": False, "complete_ending": False},
        },
    }
    (items / "a.json").write_text(json.dumps(row))
    summary = report.summarize_run("demo")
    assert summary["wer"]["synthetic"]["wer"] == 0.05
    assert summary["chapters"]["tol15"]["recall"] == 0.75
    assert (
        summary["judge"]["model_clips"]["standalone_pct"] == 100.0
        and summary["judge"]["random_windows"]["standalone_pct"] == 0.0
    )
    assert summary["cuts"]["inside_word"] == 0 and summary["cuts"]["last_word_intact_pct"] == 0.0
    assert summary["timing"]["render_seconds_per_output_minute"] == round(6 / 35 * 60, 1)


def test_caption_offset_finds_a_shifted_caption_track() -> None:
    words = [(float(t), float(t) + 0.3) for t in range(100, 400, 2)]
    cues = [(float(t) + 60, float(t) + 61.5, "one two three") for t in range(100, 400, 6)]
    assert testset.caption_offset(cues, words) == -60.0
    assert testset.caption_offset([(1.0, 2.0, "a b")], [(1.2, 1.5)]) == 0.0
