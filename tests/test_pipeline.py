"""End to end on the bundled sample with the offline model and the sidecar transcript (ffmpeg needed)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from recut.config import Settings
from recut.media.ffmpeg import probe
from recut.pipeline import ClipPatch, Pipeline, ReviewError, clip_srt
from tests.conftest import needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture()
def processed(settings: Settings, sample: Path) -> tuple[Pipeline, str]:
    pipeline = Pipeline(settings)
    project = pipeline.create(sample, name="Sample", license="CC0 1.0")
    pipeline.process(project.id)
    return pipeline, project.id


def test_every_stage_runs_and_is_saved(processed: tuple[Pipeline, str]) -> None:
    pipeline, pid = processed
    project = pipeline.store.load(pid)
    assert project.status == "ready"
    assert all(s.state in ("done", "skipped") for s in project.stages)
    assert project.source.duration > 100 and project.source.width == 1280 and project.source.sha256
    assert project.transcript is not None and len(project.transcript.speakers) == 2
    assert project.sentences and project.clips and project.chapters
    assert project.outputs.social and project.outputs.show_notes
    assert (pipeline.store.dir(pid) / "audio.wav").exists() and (pipeline.store.dir(pid) / "thumb.jpg").exists()
    for clip in project.clips:
        assert 20 <= clip.duration <= 60
        assert clip.cut_in is not None and not clip.cut_in.inside_word


def test_processing_resumes_after_a_failed_stage(settings: Settings, sample: Path) -> None:
    pipeline = Pipeline(settings)
    project = pipeline.create(sample, license="CC0")
    sidecar = Path(project.source.path).with_name(Path(project.source.path).name + ".words.json")
    sidecar.rename(sidecar.with_suffix(".bak"))
    with pytest.raises(Exception, match="sidecar"):
        pipeline.process(project.id)
    failed = pipeline.store.load(project.id)
    assert failed.stage("transcribe").state == "failed" and failed.stage("probe").state == "done"
    assert failed.status == "failed" and "transcribe" in failed.error
    sidecar.with_suffix(".bak").rename(sidecar)
    done = pipeline.process(project.id)
    assert done.status == "ready"


def test_trimming_snaps_to_word_gaps_and_invalidates_the_render(processed: tuple[Pipeline, str]) -> None:
    pipeline, pid = processed
    project = pipeline.store.load(pid)
    clip = project.clips[0]
    assert project.transcript is not None
    word = next(w for w in project.transcript.words if w.start > clip.start + 3)
    updated = pipeline.update_clip(pid, clip.id, ClipPatch(start=(word.start + word.end) / 2))
    edited = updated.clip(clip.id)
    assert edited.start <= word.start and edited.edited and edited.render is None
    assert edited.cut_in is not None and not edited.cut_in.inside_word
    assert edited.history and edited.history[-1]["before"]["start"] == clip.start


def test_too_short_trims_are_refused(processed: tuple[Pipeline, str]) -> None:
    pipeline, pid = processed
    clip = pipeline.store.load(pid).clips[0]
    with pytest.raises(ReviewError):
        pipeline.update_clip(pid, clip.id, ClipPatch(start=clip.start, end=clip.start + 1))


def test_caption_overrides_are_validated(processed: tuple[Pipeline, str]) -> None:
    pipeline, pid = processed
    project = pipeline.store.load(pid)
    clip = project.clips[0]
    assert project.transcript is not None
    inside = next(i for i, w in enumerate(project.transcript.words) if w.start >= clip.start)
    outside = next(i for i, w in enumerate(project.transcript.words) if w.start > clip.end + 1)
    updated = pipeline.update_clip(pid, clip.id, ClipPatch(caption_overrides={inside: "Fixed"}))
    assert updated.clip(clip.id).caption_overrides == {inside: "Fixed"}
    with pytest.raises(ReviewError, match="not inside"):
        pipeline.update_clip(pid, clip.id, ClipPatch(caption_overrides={outside: "x"}))
    with pytest.raises(ReviewError, match="does not exist"):
        pipeline.update_clip(pid, clip.id, ClipPatch(caption_overrides={10**6: "x"}))
    original = project.transcript.words[inside].text
    reverted = pipeline.update_clip(pid, clip.id, ClipPatch(caption_overrides={inside: original}))
    assert reverted.clip(clip.id).caption_overrides == {}


def test_approval_is_refused_with_errors_unless_forced(processed: tuple[Pipeline, str]) -> None:
    pipeline, pid = processed
    project = pipeline.store.load(pid)
    clip = project.clips[0]
    # a word-mode trim to 10 s breaks the 20 s minimum: an error-level issue
    short = pipeline.update_clip(pid, clip.id, ClipPatch(end=clip.start + 10))
    assert any(i.severity == "error" for i in short.clip(clip.id).issues)
    with pytest.raises(ReviewError, match="fix these"):
        pipeline.update_clip(pid, clip.id, ClipPatch(status="approved"))
    forced = pipeline.update_clip(pid, clip.id, ClipPatch(status="approved", force=True))
    assert forced.clip(clip.id).status == "approved"


def test_manual_clip(processed: tuple[Pipeline, str]) -> None:
    pipeline, pid = processed
    project = pipeline.add_clip(pid, 30.0, 55.0, title="My pick")
    manual = [c for c in project.clips if c.source == "manual"]
    assert len(manual) == 1 and manual[0].title == "My pick"
    assert manual[0].cut_in is not None and not manual[0].cut_in.inside_word


def test_rerunning_clips_keeps_reviewed_ones(processed: tuple[Pipeline, str]) -> None:
    pipeline, pid = processed
    project = pipeline.store.load(pid)
    keep = project.clips[0]
    pipeline.update_clip(pid, keep.id, ClipPatch(status="approved"))
    rerun = pipeline.process(pid, stages=("clips",), force=True)
    assert keep.id in {c.id for c in rerun.clips}


@pytest.mark.slow
def test_render_and_export(processed: tuple[Pipeline, str], settings: Settings) -> None:
    pipeline, pid = processed
    project = pipeline.store.load(pid)
    clip = project.clips[0]
    with pytest.raises(ReviewError, match="nothing is approved"):
        pipeline.export_bundle(pid)
    pipeline.update_clip(pid, clip.id, ClipPatch(status="approved", caption_style="karaoke"))
    with pytest.raises(ReviewError, match="render"):
        pipeline.export_bundle(pid)
    rendered = pipeline.render(pid, clip.id).clip(clip.id)
    assert rendered.render is not None and rendered.crop is not None and rendered.crop.layout == "fit"
    info = probe(Path(rendered.render.path))
    assert (info.width, info.height) == (settings.render_width, settings.render_height)
    assert abs(info.duration - rendered.duration) < 0.3
    pipeline.update_outputs(pid, {"approved": True, "social": [{"platform": "x", "approved": True}]})
    data = pipeline.export_bundle(pid)
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert any(n.endswith(".mp4") for n in names) and any(n.endswith(".srt") for n in names)
    assert "youtube/chapters.txt" in names and "social/x.txt" in names and "manifest.json" in names
    assert pipeline.store.load(pid).audit[-1]["action"] == "exported"


def test_export_needs_a_licence(settings: Settings, sample: Path) -> None:
    pipeline = Pipeline(settings)
    project = pipeline.create(sample, license="")
    pipeline.process(project.id)
    with pytest.raises(ReviewError, match="licence"):
        pipeline.export_bundle(project.id)


def test_social_posts_over_the_limit_cannot_be_approved(processed: tuple[Pipeline, str]) -> None:
    pipeline, pid = processed
    with pytest.raises(ReviewError, match="limit"):
        pipeline.update_outputs(pid, {"social": [{"platform": "x", "text": "a" * 400, "approved": True}]})
    with pytest.raises(ReviewError, match="unknown"):
        pipeline.update_outputs(pid, {"hack": 1})


def test_clip_srt(processed: tuple[Pipeline, str]) -> None:
    pipeline, pid = processed
    project = pipeline.store.load(pid)
    srt = clip_srt(project, project.clips[0])
    assert srt.startswith("1\n00:00:00,") and "-->" in srt
