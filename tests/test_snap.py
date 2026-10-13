from __future__ import annotations

import numpy as np

from recut.clips.snap import check_cut, cut_in_point, cut_out_point, fit_length, snap, snap_times
from recut.media.levels import Levels
from recut.models import Transcript
from tests.conftest import silent_levels_for


def _all_cuts_clean(result, words) -> bool:  # type: ignore[no-untyped-def]
    return not any(w.start < result.start < w.end or w.start < result.end < w.end for w in words)


def test_cut_points_fall_in_the_gaps(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    levels = silent_levels_for(transcript.words)
    result = snap(0, 3, sentences, transcript.words, levels, min_s=5, max_s=60, pad_in=0.12, pad_out=0.3)
    first = transcript.words[sentences[0].first_word]
    last = transcript.words[sentences[result.last_sentence].last_word]
    assert result.start <= first.start
    assert result.end >= last.end
    assert _all_cuts_clean(result, transcript.words)
    assert not result.cut_in.inside_word and not result.cut_out.inside_word
    assert result.cut_out.level_db is not None and result.cut_out.level_db < -30


def test_cut_out_never_reaches_the_next_word(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    end_word = sentences[2].last_word
    t = cut_out_point(end_word, transcript.words, None, pad=5.0, media_end=None)
    assert transcript.words[end_word].end < t < transcript.words[end_word + 1].start


def test_cut_in_never_reaches_the_previous_word(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    first = sentences[3].first_word
    t = cut_in_point(first, transcript.words, None, pad=5.0)
    assert transcript.words[first - 1].end < t < transcript.words[first].start


def test_quietest_point_is_preferred() -> None:
    db = np.full(300, -20.0, dtype=np.float32)
    db[140:150] = -80.0
    levels = Levels(db)
    assert 1.38 <= levels.quietest(1.0, 2.0) <= 1.5


def test_fit_length_extends_short_ranges_by_whole_sentences(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    _, sentences = talk
    first, last, notes = fit_length(0, 0, sentences, min_s=20, max_s=60)
    assert first == 0 and last > 0
    assert sentences[last].end - sentences[first].start >= 20
    assert notes and "extended" in notes[0]


def test_fit_length_shortens_long_ranges_and_keeps_the_hook(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    _, sentences = talk
    first, last, notes = fit_length(0, len(sentences) - 1, sentences, min_s=20, max_s=30)
    assert first == 0
    assert sentences[last].end - sentences[first].start <= 30
    assert "shortened" in notes[0]


def test_fit_length_clamps_out_of_range_indices(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    _, sentences = talk
    first, last, _ = fit_length(-5, 999, sentences, min_s=0, max_s=10_000)
    assert first == 0 and last == len(sentences) - 1


def test_check_cut_detects_a_mid_word_cut(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    transcript, _ = talk
    w = transcript.words[5]
    inside = check_cut((w.start + w.end) / 2, transcript.words, None)
    assert inside.inside_word and inside.gap_before < 0
    gap = check_cut(w.end + 0.04, transcript.words, None)
    assert not gap.inside_word and gap.gap_before > 0 and gap.gap_after > 0


def test_snap_times_to_sentences_widens_to_whole_sentences(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    mid_second = (sentences[1].start + sentences[1].end) / 2
    mid_fourth = (sentences[4].start + sentences[4].end) / 2
    result = snap_times(
        mid_second, mid_fourth, sentences, transcript.words, None, min_s=0, max_s=600, pad_in=0.1, pad_out=0.2
    )
    assert result.first_sentence == 1 and result.last_sentence == 4
    assert result.start < sentences[1].start and result.end > sentences[4].end


def test_snap_times_word_mode_keeps_partial_sentences_but_whole_words(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    transcript, sentences = talk
    w = transcript.words[sentences[1].first_word + 2]
    result = snap_times(
        (w.start + w.end) / 2,
        sentences[3].end,
        sentences,
        transcript.words,
        None,
        min_s=0,
        max_s=600,
        pad_in=0.1,
        pad_out=0.2,
        to_sentences=False,
    )
    assert result.start < w.start  # the word that was cut through is kept whole
    assert "starts inside a sentence" in result.adjustments
    assert _all_cuts_clean(result, transcript.words)


def test_cut_out_at_the_end_of_the_media(talk: tuple[Transcript, list]) -> None:  # type: ignore[type-arg]
    transcript, _sentences = talk
    last = len(transcript.words) - 1
    t = cut_out_point(last, transcript.words, None, pad=0.3, media_end=transcript.words[-1].end + 0.1)
    assert transcript.words[-1].end < t <= transcript.words[-1].end + 0.1
