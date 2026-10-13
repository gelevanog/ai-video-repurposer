from __future__ import annotations

from recut.clips.lint import lint_clip, penalty
from recut.models import ClipIssue, CutCheck
from recut.text import sentences_from_words
from tests.conftest import make_words


def _lint(text: str, *, min_s: float = 0, max_s: float = 600, **kw: object) -> list[str]:
    words = make_words(text)
    sentences = sentences_from_words(words)
    issues = lint_clip(
        0, len(sentences) - 1, sentences, words, start=0, end=words[-1].end, min_s=min_s, max_s=max_s, **kw
    )  # type: ignore[arg-type]
    return [i.code for i in issues]


def test_clean_clip_has_no_issues() -> None:
    assert _lint("Most creators charge too little. Price the result, not the hours.") == []


def test_connector_start_is_flagged() -> None:
    assert "starts_with_connector" in _lint("And then we raised prices. It worked.")


def test_pronoun_start_is_a_dangling_reference() -> None:
    assert "dangling_reference" in _lint("He told me to double it. I did.")


def test_self_introducing_this_is_fine() -> None:
    assert "dangling_reference" not in _lint("This is the biggest mistake creators make. Do not do it.")
    assert "dangling_reference" in _lint("That changed everything for us. We grew.")


def test_context_phrases_are_flagged() -> None:
    assert "context_reference" in _lint("Pricing matters. As I said, hourly rates are a trap.")


def test_ending_on_a_question_or_mid_sentence() -> None:
    assert "ends_on_question" in _lint("Pricing matters. But what should you charge?")
    assert "ends_mid_sentence" in _lint("Pricing matters. And the next thing is")


def test_greeting_is_a_weak_hook() -> None:
    assert "weak_hook" in _lint("Welcome back to the show. Today we talk money.")


def test_length_outside_limits_is_an_error() -> None:
    assert "length" in _lint("Short clip here.", min_s=20, max_s=60)


def test_cut_findings() -> None:
    bad = CutCheck(at=1.0, inside_word=True, gap_before=-0.1, gap_after=-0.1, level_db=-5)
    tight = CutCheck(at=1.0, inside_word=False, gap_before=0.01, gap_after=0.5, level_db=-40)
    codes = _lint("Most creators charge too little.", cut_in=bad, cut_out=tight)
    assert {"cut_in_mid_word", "cut_in_loud", "cut_out_tight"} <= set(codes)


def test_fillers_and_many_speakers_are_info() -> None:
    words = make_words("um so uh this um is uh it.")
    for i, w in enumerate(words):
        w.speaker = f"S{i % 3 + 1}"
    sentences = sentences_from_words(words)
    issues = lint_clip(0, len(sentences) - 1, sentences, words, start=0, end=words[-1].end, min_s=0, max_s=99)
    codes = {i.code for i in issues}
    assert "fillers" in codes and "many_speakers" in codes


def test_penalty_weights_errors_most() -> None:
    issues = [
        ClipIssue(code="a", severity="error", message=""),
        ClipIssue(code="b", severity="warning", message=""),
        ClipIssue(code="c", severity="info", message=""),
    ]
    assert penalty(issues) == 25 + 6 + 1
