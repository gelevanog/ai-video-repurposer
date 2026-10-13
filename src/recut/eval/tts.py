"""Kokoro-82M text to speech (ONNX, CPU, Apache-2.0) for the synthetic test talks. Used only to build the test set;
the product itself never synthesizes speech."""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

MODEL_DIR = Path.home() / ".cache" / "recut" / "models"
TTS_RATE = 24000


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str


KOKORO_FILES = [
    ModelFile(
        "kokoro-v1.0.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    ),
    ModelFile(
        "voices-v1.0.bin",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    ),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def ensure_models(directory: Path = MODEL_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for f in KOKORO_FILES:
        path = directory / f.name
        if path.exists() and _sha256(path) == f.sha256:
            continue
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f.url, tmp)  # noqa: S310 - fixed https URL
        if _sha256(tmp) != f.sha256:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for {f.name}")
        tmp.replace(path)
    return directory


class Voice:
    _lock = threading.Lock()
    _engine: Any = None

    def __init__(self, directory: Path = MODEL_DIR) -> None:
        with Voice._lock:
            if Voice._engine is None:
                from kokoro_onnx import Kokoro

                ensure_models(directory)
                Voice._engine = Kokoro(str(directory / "kokoro-v1.0.onnx"), str(directory / "voices-v1.0.bin"))
        self.engine = Voice._engine

    def say(self, text: str, voice: str, *, speed: float = 1.0) -> npt.NDArray[np.float32]:
        lang = "en-gb" if voice.startswith("b") else "en-us"
        samples, rate = self.engine.create(text, voice=voice, speed=speed, lang=lang)
        audio = np.asarray(samples, dtype=np.float32)
        if rate != TTS_RATE:
            raise RuntimeError(f"unexpected Kokoro sample rate {rate}")
        return trim_silence(audio)


def trim_silence(audio: npt.NDArray[np.float32], threshold: float = 0.01) -> npt.NDArray[np.float32]:
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if len(loud) == 0:
        return audio
    pad = int(0.03 * TTS_RATE)
    return audio[max(0, int(loud[0]) - pad) : min(len(audio), int(loud[-1]) + pad)]
