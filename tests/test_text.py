from __future__ import annotations

from recut.models import Word
from recut.text import is_sentence_end, normalize_for_wer, sentences_from_words, ts, wer, word_count
from tests.conftest import make_words


def test_sentences_split_at_terminal_punctuation() -> None:
    words = make_words("Hello there. How are you? Fine!")
    sentences = sentences_from_words(words)
    assert [s.text for s in sentences] == ["Hello there.", "How are you?", "Fine!"]
    assert sentences[1].first_word == 2 and sentences[1].last_word == 4
    assert sentences[0].start == words[0].start and sentences[2].end == words[-1].end


def test_abbreviations_and_initialisms_do_not_end_sentences() -> None:
    assert not is_sentence_end("Dr.")
    assert not is_sentence_end("U.S.")
    assert not is_sentence_end("e.g.")
    assert is_sentence_end("done.")
    assert is_sentence_end('said."')


def test_speaker_change_starts_a_new_sentence() -> None:
    words = make_words("so what do you think", speaker="S1") + make_words("I think so.", start=5.0, speaker="S2")
    sentences = sentences_from_words(words)
    assert [s.speaker for s in sentences] == ["S1", "S2"]


def test_long_run_on_is_split_at_a_pause() -> None:
    text = " ".join(["word"] * 60)
    words = make_words(text)
    words[50].end = words[50].start + 0.2
    for w in words[51:]:
        w.start += 1.0
        w.end += 1.0
    sentences = sentences_from_words(words)
    assert len(sentences) >= 2


def test_timestamp_formatting() -> None:
    assert ts(0) == "00:00"
    assert ts(65.4) == "01:05"
    assert ts(3725) == "1:02:05"
    assert ts(65.47, millis=True) == "01:05.4"
    assert ts(-3) == "00:00"


def test_wer_normalization_ignores_formatting() -> None:
    assert normalize_for_wer("It's 10% — ten percent! Don't [applause]") == ["its", "10", "percent", "10", "percent", "do", "not"]
    assert normalize_for_wer("Um, uh, so") == ["so"]
    assert normalize_for_wer("$40,000") == ["40000"]


def test_wer_counts_substitutions_deletions_insertions() -> None:
    result = wer("the cat sat on the mat", "the cat sat on a mat today")
    assert result["substitutions"] == 1 and result["insertions"] == 1 and result["deletions"] == 0
    assert abs(result["wer"] - 2 / 6) < 1e-9
    assert wer("same words here", "Same words, here.")["wer"] == 0.0


def test_wer_of_empty_reference_does_not_divide_by_zero() -> None:
    assert wer("", "something")["wer"] == 1.0


def test_word_count() -> None:
    assert word_count("Don't stop - it's 2026, okay?") == 5


def test_word_model_defaults() -> None:
    w = Word(start=0, end=1, text="hi")
    assert w.speaker == "S1" and w.prob == 1.0
