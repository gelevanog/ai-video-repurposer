"""Real ffmpeg renders of generated media: the tracked crop expression, the centre crop and the audiogram."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from recut.config import Settings
from recut.media.ffmpeg import MediaError, extract_audio, load_wav, probe, sample_frames, sha256_file, thumbnail
from recut.media.levels import Levels
from recut.models import Clip, CropKey, CropPlan
from recut.render.render import RenderJob, build_command, escape_filter_path, plan_crop, render_clip
from tests.conftest import make_words, needs_ffmpeg

pytestmark = [needs_ffmpeg, pytest.mark.slow]


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("media") / "bars.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=25:duration=6",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=330:duration=6",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    )
    return path


@pytest.fixture(scope="module")
def audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("media") / "voice.m4a"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=220:duration=5", "-c:a", "aac", str(path))
    return path


def _clip(start: float = 1.0, end: float = 5.0, **kw: object) -> Clip:
    return Clip(id="clip_t", start=start, end=end, first_sentence=0, last_sentence=0, title="A test clip title", **kw)  # type: ignore[arg-type]


def test_probe_and_audio_extraction(video: Path, tmp_path: Path) -> None:
    info = probe(video)
    assert (info.width, info.height, info.has_audio) == (640, 360, True) and 5.9 < info.duration < 6.2
    wav = extract_audio(video, tmp_path / "a.wav", start=1.0, end=3.0)
    samples = load_wav(wav)
    assert abs(len(samples) / 16000 - 2.0) < 0.05
    levels = Levels.from_samples(samples)
    assert levels.speech_level() > -30 and abs(levels.duration - 2.0) < 0.05
    assert len(sha256_file(video)) == 64


def test_probe_errors(tmp_path: Path) -> None:
    with pytest.raises(MediaError):
        probe(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(MediaError):
        probe(junk)


def test_frames_and_thumbnail(video: Path, tmp_path: Path) -> None:
    frames = list(sample_frames(video, start=0, end=2, fps=2, width=320, src_width=640, src_height=360))
    assert len(frames) in (4, 5) and frames[0][1].shape == (180, 320, 3)
    assert thumbnail(video, tmp_path / "t.jpg", at=1.0).stat().st_size > 1000


def test_tracked_crop_renders(video: Path, tmp_path: Path, settings: Settings) -> None:
    plan = CropPlan(layout="track", keys=[CropKey(t=0, cx=0.3), CropKey(t=1.5, cx=0.3), CropKey(t=3.0, cx=0.7)])
    words = make_words("one two three four five six seven", start=1.0)
    info = render_clip(
        video, probe(video), _clip(caption_style="pop"), words, tmp_path / "track.mp4", settings, plan=plan
    )
    out = probe(tmp_path / "track.mp4")
    assert (out.width, out.height) == (settings.render_width, settings.render_height)
    assert abs(out.duration - 4.0) < 0.2 and info.layout == "track" and info.seconds > 0
    assert (tmp_path / "track.ass").exists()


def test_auto_layout_without_faces_is_fit(video: Path, settings: Settings) -> None:
    plan = plan_crop(video, probe(video), 0, 3, settings, layout="auto")
    assert plan.layout == "fit" and plan.samples >= 8 and plan.face_ratio == 0


def test_audio_only_source_becomes_an_audiogram(audio: Path, tmp_path: Path, settings: Settings) -> None:
    info = probe(audio)
    assert not info.has_video
    words = make_words("an audio only podcast clip", start=0.5)
    result = render_clip(audio, info, _clip(start=0.0, end=4.0), words, tmp_path / "wave.mp4", settings)
    out = probe(tmp_path / "wave.mp4")
    assert out.has_video and (out.width, out.height) == (settings.render_width, settings.render_height)
    assert result.layout == "fit"


def test_center_layout_and_command_shape(video: Path, tmp_path: Path, settings: Settings) -> None:
    job = RenderJob(
        video, tmp_path / "o.mp4", 1.0, 3.0, probe(video), CropPlan(layout="center"), tmp_path / "x.ass", 360, 640
    )
    cmd = build_command(job, settings)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "crop=w=202:h=360:x='219'" in graph and "ass=filename=" in graph
    assert cmd[cmd.index("-ss") + 1] == "1.000" and cmd[cmd.index("-t") + 1] == "2.000"


def test_filter_paths_are_escaped(tmp_path: Path) -> None:
    weird = tmp_path / "it's: a file.ass"
    assert "\\'" in escape_filter_path(weird) and "\\:" in escape_filter_path(weird)
