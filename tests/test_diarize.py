from __future__ import annotations

import numpy as np

from recut.asr.diarize import apply, chunk_features, chunk_words, diarize
from recut.media.ffmpeg import SAMPLE_RATE
from recut.models import Transcript, Word

RNG = np.random.default_rng(3)


def _voice(seconds: float, f0: float, bright: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
    sig = sum((bright**k) * np.sin(2 * np.pi * f0 * k * vibrato * t) for k in range(1, 12))
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2
    return (0.2 * sig * envelope / 4 + 0.003 * RNG.standard_normal(len(t))).astype(np.float32)


def _conversation(turns: list[tuple[str, float, float]]) -> tuple[Transcript, np.ndarray]:
    """Turns of (speaker, f0, brightness), 2.5 s each with 0.6 s pauses; 5 words per turn."""
    audio: list[np.ndarray] = []
    words: list[Word] = []
    t = 0.0
    gold = []
    for speaker, f0, bright in turns:
        seg = _voice(2.5, f0, bright)
        for k in range(5):
            words.append(Word(start=round(t + k * 0.5, 3), end=round(t + k * 0.5 + 0.42, 3), text=f"w{k}"))
            gold.append(speaker)
        audio.append(seg)
        audio.append(np.zeros(int(0.6 * SAMPLE_RATE), dtype=np.float32))
        t += 3.1
    transcript = Transcript(duration=t, model="test", words=words)
    transcript.__dict__["gold"] = gold
    return transcript, np.concatenate(audio)


def test_chunks_split_at_pauses_and_length() -> None:
    words = [Word(start=i * 0.5, end=i * 0.5 + 0.4, text="w") for i in range(30)]
    words[10].start += 1.0
    chunks = chunk_words(words, pause=0.35, max_len=6.0)
    assert chunks[0].last == 9
    assert all(c.duration <= 6.5 for c in chunks)
    assert chunk_words([]) == []


def test_features_need_enough_speech() -> None:
    assert chunk_features(np.zeros(100, dtype=np.float32)) is None
    f = chunk_features(_voice(1.5, 120, 0.7))
    assert f is not None and f.shape == (40,)


def test_two_clearly_different_voices_are_separated() -> None:
    turns = [("A", 110, 0.55), ("B", 230, 0.8)] * 6
    transcript, audio = _conversation(turns)
    result = diarize(transcript, audio, max_speakers=4)
    assert result.method == "mfcc-cluster" and len(result.speakers) == 2
    gold = transcript.__dict__["gold"]
    mapping = {result.labels[0]: gold[0]}
    other = next(label for label in result.labels if label != result.labels[0])
    mapping[other] = next(g for g in gold if g != gold[0])
    accuracy = sum(mapping[p] == g for p, g in zip(result.labels, gold, strict=True)) / len(gold)
    assert accuracy >= 0.9
    assert result.labels[0] == "S1"  # the first voice is S1


def test_one_voice_stays_one_speaker() -> None:
    transcript, audio = _conversation([("A", 120, 0.6)] * 12)
    result = diarize(transcript, audio, max_speakers=4)
    assert result.speakers == ["S1"] and set(result.labels) == {"S1"}
    assert "single speaker" in result.note


def test_too_little_speech_is_single_speaker() -> None:
    transcript, audio = _conversation([("A", 120, 0.6), ("B", 230, 0.8)])
    result = diarize(transcript, audio)
    assert result.method == "single" and "too little speech" in result.note


def test_apply_writes_speakers_onto_words() -> None:
    turns = [("A", 110, 0.55), ("B", 230, 0.8)] * 6
    transcript, audio = _conversation(turns)
    out = apply(transcript, diarize(transcript, audio))
    assert out.diarization == "mfcc-cluster" and out.speakers == ["S1", "S2"]
    assert {w.speaker for w in out.words} == {"S1", "S2"}
