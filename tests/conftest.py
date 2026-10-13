"""Shared fixtures: offline settings (fake model, sidecar transcripts, no ledger, no cache) and transcript builders."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from recut.config import REPO_ROOT, Settings
from recut.media.levels import Levels
from recut.models import Transcript, Word
from recut.text import sentences_from_words

SAMPLE = REPO_ROOT / "data" / "samples" / "creator-margin-ep12.mp4"
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "projects",
        llm_provider="fake",
        asr="sidecar",
        llm_ledger=None,
        llm_cache=False,
        transcript_cache_dir=None,
        seed_demo=False,
        render_width=360,
        render_height=640,
        render_preset="ultrafast",
        loudnorm=False,
        llm_min_seconds_between_requests=0.0,
    )


@pytest.fixture()
def sample(tmp_path: Path) -> Path:
    """A copy of the bundled sample (and its sidecar transcript) in a temp folder."""
    target = tmp_path / "media" / SAMPLE.name
    target.parent.mkdir(parents=True)
    shutil.copy2(SAMPLE, target)
    shutil.copy2(SAMPLE.with_name(SAMPLE.name + ".words.json"), target.with_name(target.name + ".words.json"))
    return target


def make_words(
    text: str,
    *,
    start: float = 0.0,
    word: float = 0.3,
    gap: float = 0.08,
    sentence_gap: float = 0.6,
    speaker: str = "S1",
) -> list[Word]:
    """Evenly timed words; a longer pause after each sentence end."""
    out: list[Word] = []
    t = start
    for token in text.split():
        out.append(Word(start=round(t, 3), end=round(t + word, 3), text=token, speaker=speaker))
        t += word + (sentence_gap if token.endswith((".", "?", "!")) else gap)
    return out


def make_transcript(text: str, **kwargs: float) -> Transcript:
    words = make_words(text, **kwargs)  # type: ignore[arg-type]
    return Transcript(duration=words[-1].end + 1.0, model="test", words=words)


def silent_levels_for(words: list[Word], duration: float | None = None) -> Levels:
    """Synthetic levels: -20 dB inside words, -70 dB in the gaps."""
    end = duration or (words[-1].end + 1.0)
    n = int(end / 0.01) + 1
    db = np.full(n, -70.0, dtype=np.float32)
    for w in words:
        db[int(w.start / 0.01) : int(w.end / 0.01) + 1] = -20.0
    return Levels(db)


LONG_TALK = " ".join(
    [
        "Here is the truth about pricing that nobody tells you.",
        "Most creators charge far too little for their work.",
        "I once charged 200 dollars for a video that made a client 40,000 dollars.",
        "The video was not the problem, my price was.",
        "So the lesson is simple: price the result, not the hours.",
        "And that brings us to the second point.",
        "As I said, hourly rates punish you for getting faster.",
        "Why do people still use them?",
        "Because they feel safe and easy to explain.",
        "But safe is expensive in the long run.",
        "Here is the three number test I use before every proposal.",
        "Write down your floor, your target and your stretch price.",
        "Then quote the stretch number first and watch what happens.",
        "Half the time the client says yes and you learn what you are worth.",
        "That is the whole trick.",
    ]
)


@pytest.fixture()
def talk() -> Iterator[tuple[Transcript, list]]:  # type: ignore[type-arg]
    transcript = make_transcript(LONG_TALK)
    yield transcript, sentences_from_words(transcript.words)
