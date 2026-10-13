"""Speech to text on CPU with faster-whisper (CTranslate2, int8) and word timestamps.

`small` is the default: on the evaluation machine it transcribes about 10x faster than real time on 16 cores and is
markedly more accurate on names and numbers than `base`. The sidecar backend reads a transcript saved next to the
media file instead (`<file>.words.json`): CI, the tests and the zero-download demo use it."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from recut.config import Settings
from recut.logging_config import get_logger
from recut.media.ffmpeg import load_wav
from recut.models import Transcript, Word

log = get_logger(__name__)

_MODELS: dict[tuple[str, str, int], Any] = {}
_LOCK = threading.Lock()


class ASRUnavailableError(RuntimeError):
    pass


def sidecar_path(media: Path) -> Path:
    return media.with_name(media.name + ".words.json")


def load_sidecar(media: Path) -> Transcript:
    path = sidecar_path(media)
    if not path.exists():
        raise ASRUnavailableError(
            f"no sidecar transcript {path.name}: install the `asr` extra (faster-whisper) or set RECUT_ASR=whisper"
        )
    transcript = Transcript.model_validate_json(path.read_text(encoding="utf-8"))
    return transcript.model_copy(update={"model": transcript.model or "sidecar"})


def save_sidecar(media: Path, transcript: Transcript) -> Path:
    path = sidecar_path(media)
    path.write_text(transcript.model_dump_json(indent=None), encoding="utf-8")
    return path


def _model(settings: Settings) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ASRUnavailableError("faster-whisper is not installed: `uv sync --extra asr`") from exc
    threads = settings.whisper_threads or (os.cpu_count() or 4)
    key = (settings.whisper_model, settings.whisper_compute_type, threads)
    with _LOCK:
        if key not in _MODELS:
            log.info("asr.load", model=settings.whisper_model, compute_type=settings.whisper_compute_type)
            _MODELS[key] = WhisperModel(
                settings.whisper_model, device="cpu", compute_type=settings.whisper_compute_type, cpu_threads=threads
            )
        return _MODELS[key]


def transcribe_wav(wav: Path, settings: Settings, *, duration: float | None = None) -> Transcript:
    model = _model(settings)
    started = time.monotonic()
    # decoded here (not by faster-whisper's PyAV path): a float32 array at 16 kHz
    audio = load_wav(wav)
    segments, info = model.transcribe(
        audio,
        language=settings.whisper_language or None,
        beam_size=settings.whisper_beam_size,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    for segment in segments:
        for w in segment.words or []:
            text = w.word.strip()
            if not text:
                continue
            start, end = float(w.start), float(w.end)
            if end <= start:
                end = start + 0.05
            words.append(Word(start=round(start, 3), end=round(end, 3), text=text, prob=round(float(w.probability), 3)))
    words = _monotonic(words)
    elapsed = time.monotonic() - started
    total = duration if duration is not None else float(getattr(info, "duration", 0.0) or 0.0)
    log.info("asr.done", words=len(words), seconds=round(elapsed, 1), audio_seconds=round(total, 1))
    return Transcript(
        language=str(getattr(info, "language", settings.whisper_language) or "en"),
        duration=round(total, 3),
        model=f"faster-whisper {settings.whisper_model} ({settings.whisper_compute_type}, CPU)",
        words=words,
        seconds=round(elapsed, 2),
    )


def _monotonic(words: list[Word]) -> list[Word]:
    """Whisper occasionally emits overlapping word times; clamp so that word i ends before word i+1 starts."""
    for i in range(len(words) - 1):
        nxt = words[i + 1]
        if words[i].end > nxt.start:
            mid = round((words[i].end + nxt.start) / 2, 3)
            if mid <= words[i].start:
                mid = round(words[i].start + 0.01, 3)
            words[i].end = mid
            if nxt.start < mid:
                nxt.start = mid
            if nxt.end <= nxt.start:
                nxt.end = round(nxt.start + 0.01, 3)
    return words


def transcribe(media: Path, wav: Path, settings: Settings, *, duration: float | None = None) -> Transcript:
    if settings.asr == "sidecar":
        return load_sidecar(media)
    if sidecar_path(media).exists() and os.environ.get("RECUT_PREFER_SIDECAR") == "1":
        return load_sidecar(media)
    return transcribe_wav(wav, settings, duration=duration)
