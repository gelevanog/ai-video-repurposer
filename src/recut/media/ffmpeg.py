"""Thin, typed wrappers around ffmpeg and ffprobe: probing, audio extraction, frame sampling.

Every command is an argument list (never a shell string), so file names cannot inject options or commands."""

from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16000


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class Probe:
    duration: float
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool
    size_bytes: int


def run(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise MediaError(f"{cmd[0]} is not installed (needed for media processing)") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{Path(cmd[0]).name} timed out after {timeout} s") from exc
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise MediaError(f"{Path(cmd[0]).name} failed ({proc.returncode}): " + " | ".join(tail))
    return proc


def _fps(rate: str | None) -> float:
    if not rate or rate in ("0/0", "0"):
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(rate)


def probe(path: Path, *, ffprobe: str = "ffprobe") -> Probe:
    if not path.exists():
        raise MediaError(f"no such file: {path}")
    proc = run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "--", str(path)],
        timeout=60,
    )
    data = json.loads(proc.stdout or b"{}")
    streams = data.get("streams") or []
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None and video is None:
        raise MediaError(f"{path.name}: no audio or video stream")
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if not duration and audio is not None:
        duration = float(audio.get("duration") or 0.0)
    return Probe(
        duration=round(duration, 3),
        width=int(video.get("width") or 0) if video else 0,
        height=int(video.get("height") or 0) if video else 0,
        fps=round(_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")), 3) if video else 0.0,
        has_video=video is not None,
        has_audio=audio is not None,
        size_bytes=path.stat().st_size,
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def extract_audio(src: Path, dst: Path, *, ffmpeg: str = "ffmpeg", start: float = 0.0, end: float | None = None) -> Path:
    """16 kHz mono 16-bit WAV, the input of speech recognition, diarization and silence detection."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tmp)]
    run(cmd, timeout=3600)
    tmp.replace(dst)
    return dst


def load_wav(path: Path) -> npt.NDArray[np.float32]:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise MediaError(f"{path.name}: expected 16-bit PCM")
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def save_wav(path: Path, samples: npt.NDArray[np.float32], rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


def sample_frames(
    src: Path,
    *,
    start: float,
    end: float,
    fps: float = 3.0,
    width: int = 480,
    src_width: int,
    src_height: int,
    ffmpeg: str = "ffmpeg",
) -> Iterator[tuple[float, npt.NDArray[np.uint8]]]:
    """Low-resolution BGR frames between start and end, `fps` per second, decoded by ffmpeg into a pipe."""
    height = max(2, round(src_height * width / max(1, src_width) / 2) * 2)
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(src),
        "-an",
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    frame_bytes = width * height * 3
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise MediaError(f"{ffmpeg} is not installed") from exc
    assert proc.stdout is not None
    index = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            yield index / fps, frame
            index += 1
    finally:
        proc.stdout.close()
        proc.wait(timeout=30)


def thumbnail(src: Path, dst: Path, *, at: float, width: int = 640, ffmpeg: str = "ffmpeg") -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, at):.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "4",
            str(dst),
        ],
        timeout=120,
    )
    return dst
