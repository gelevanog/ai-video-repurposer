"""Synthetic talks and podcasts for the test set: a script (chapters of speaker turns, one planted highlight per
chapter) becomes a 1280x720 slide video with Kokoro voices. Because the audio is generated, the gold data is exact:
the transcript, every sentence's time, the speaker of every sentence, the chapter starts and the planted
highlights. The scripts were drafted by a free model from outlines and are committed (data/testset/synthetic/)."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from recut.config import ASSETS_DIR
from recut.eval.tts import TTS_RATE, Voice
from recut.media.ffmpeg import run

W, H = 1280, 720
SENTENCE_PAUSE = 0.28
TURN_PAUSE = 0.65
CHAPTER_PAUSE = 1.2
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“'$])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text.strip()) if s.strip()]


@dataclass
class GoldSentence:
    start: float
    end: float
    text: str
    speaker: str
    chapter: int
    turn: int
    highlight: bool


@dataclass
class Gold:
    id: str
    title: str
    duration: float
    sentences: list[GoldSentence] = field(default_factory=list)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    highlights: list[dict[str, Any]] = field(default_factory=list)
    speakers: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "duration": round(self.duration, 3),
            "speakers": self.speakers,
            "chapters": self.chapters,
            "highlights": self.highlights,
            "sentences": [s.__dict__ for s in self.sentences],
            "transcript": " ".join(s.text for s in self.sentences),
        }


def _font(size: int, *, bold: bool = True) -> Any:
    from PIL import ImageFont

    candidates = [ASSETS_DIR / "fonts" / "Poppins-ExtraBold.ttf"] if bold else []
    candidates += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/TTF/DejaVuSans.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _wrap(draw: Any, text: str, font: Any, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=font) > width and line:
            lines.append(line)
            line = w
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def slide(path: Path, *, show: str, title: str, bullets: list[str], speaker: str, accent: tuple[int, int, int]) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), (16, 18, 28))
    draw = ImageDraw.Draw(img)
    for y in range(H):  # soft vertical gradient
        shade = int(10 * y / H)
        draw.line([(0, y), (W, y)], fill=(16 + shade, 18 + shade, 30 + shade * 2))
    draw.rectangle([0, 0, 14, H], fill=accent)
    draw.text((70, 48), show.upper(), font=_font(22), fill=accent)
    y = 110
    for line in _wrap(draw, title, _font(58), W - 160)[:2]:
        draw.text((70, y), line, font=_font(58), fill=(245, 246, 250))
        y += 74
    y += 28
    for bullet in bullets[:4]:
        lines = _wrap(draw, bullet, _font(30, bold=False), W - 220)[:2]
        draw.ellipse([78, y + 13, 92, y + 27], fill=accent)
        for line in lines:
            draw.text((112, y), line, font=_font(30, bold=False), fill=(200, 204, 218))
            y += 42
        y += 18
    tag = f"  {speaker}  "
    tw = draw.textlength(tag, font=_font(26))
    draw.rounded_rectangle([70, H - 92, 70 + tw + 40, H - 44], radius=12, fill=(36, 40, 58))
    draw.ellipse([84, H - 76, 100, H - 60], fill=(239, 68, 68))
    draw.text((104, H - 86), tag, font=_font(26), fill=(235, 236, 242))
    draw.text((W - 330, H - 80), "synthetic test media · CC0", font=_font(18, bold=False), fill=(110, 114, 132))
    img.save(path)
    return path


def build(script: dict[str, Any], out_dir: Path, *, voice: Voice | None = None) -> tuple[Path, Gold]:
    """Render <out_dir>/<id>.mp4 and <id>.gold.json from a script."""
    voice = voice or Voice()
    sid = script["id"]
    speakers: dict[str, dict[str, str]] = script["speakers"]
    accent = tuple(script.get("accent", [124, 92, 255]))
    gold = Gold(id=sid, title=script["title"], duration=0.0, speakers={k: v["name"] for k, v in speakers.items()})
    audio: list[npt.NDArray[np.float32]] = []
    t = 0.0
    segments: list[tuple[float, float, dict[str, Any], str, int]] = []  # (start, end, chapter, speaker, ci)

    def silence(seconds: float) -> None:
        nonlocal t
        audio.append(np.zeros(int(seconds * TTS_RATE), dtype=np.float32))
        t += seconds

    silence(0.6)
    for ci, chapter in enumerate(script["chapters"]):
        if ci:
            silence(CHAPTER_PAUSE)
        chapter_start = t
        gold.chapters.append({"start": round(chapter_start, 3), "title": chapter["title"]})
        for ti, turn in enumerate(chapter["turns"]):
            if ti:
                silence(TURN_PAUSE)
            spk = turn["speaker"]
            turn_start = t
            first_sentence = len(gold.sentences)
            for si, sentence in enumerate(split_sentences(turn["text"])):
                if si:
                    silence(SENTENCE_PAUSE)
                samples = voice.say(sentence, speakers[spk]["voice"], speed=float(speakers[spk].get("speed", 1.0)))
                start = t
                audio.append(samples)
                t += len(samples) / TTS_RATE
                gold.sentences.append(
                    GoldSentence(round(start, 3), round(t, 3), sentence, spk, ci, ti, bool(turn.get("highlight")))
                )
            segments.append((turn_start, t, chapter, speakers[spk]["name"], ci))
            if turn.get("highlight"):
                gold.highlights.append(
                    {
                        "chapter": ci,
                        "start": round(turn_start, 3),
                        "end": round(t, 3),
                        "sentences": [first_sentence, len(gold.sentences) - 1],
                        "text": turn["text"],
                    }
                )
    silence(1.0)
    gold.duration = t
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = np.concatenate(audio)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wav_path = tmp / "audio.wav"
        _write_wav(wav_path, wav)
        concat = []
        # slide per turn; the gaps between turns stay on the previous slide
        bounds = [s[0] for s in segments[1:]] + [t]
        prev_end = 0.0
        for i, (start, _end, chapter, name, _ci) in enumerate(segments):
            img = slide(
                tmp / f"s{i:03d}.png",
                show=script.get("show", script["title"]),
                title=chapter["title"],
                bullets=chapter.get("bullets", []),
                speaker=name,
                accent=accent,
            )
            duration = bounds[i] - (prev_end if i else 0.0)
            prev_end = bounds[i]
            concat.append(f"file '{img}'\nduration {duration:.3f}")
            _ = start
        concat.append(f"file '{tmp / f's{len(segments) - 1:03d}.png'}'")
        (tmp / "list.txt").write_text("\n".join(concat) + "\n")
        video = out_dir / f"{sid}.mp4"
        run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(tmp / "list.txt"),
                "-i",
                str(wav_path),
                "-vf",
                "fps=25,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-tune",
                "stillimage",
                "-crf",
                "28",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(video),
            ],
            timeout=3600,
        )
    (out_dir / f"{sid}.gold.json").write_text(json.dumps(gold.to_json(), indent=1, ensure_ascii=False))
    return video, gold


def _write_wav(path: Path, audio: npt.NDArray[np.float32]) -> None:
    import wave

    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TTS_RATE)
        handle.writeframes(pcm.tobytes())


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        check=True,
    )
    return float(proc.stdout.strip() or 0)
