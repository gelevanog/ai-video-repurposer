"""The end-to-end flow and every review action, shared by the API, the CLI, the MCP server and the evaluation.

probe -> audio -> transcribe -> diarize -> clips -> chapters -> posts; then, per clip, the reviewer edits, approves
and renders. Every stage saves the project, so a crash or a restart resumes from the last finished stage."""

from __future__ import annotations

import io
import json
import shutil
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from recut.asr import diarize as diarization
from recut.asr.whisper import transcribe
from recut.chapters import format_youtube, make_chapters, youtube_ready
from recut.clips.lint import lint_clip, penalty
from recut.clips.snap import snap_times
from recut.clips.suggest import rank_score, suggest
from recut.config import Settings
from recut.content.posts import generate
from recut.llm.base import ChatModel
from recut.llm.factory import model_for
from recut.logging_config import get_logger
from recut.media.ffmpeg import MediaError, Probe, extract_audio, load_wav, probe, sha256_file, thumbnail
from recut.media.levels import Levels
from recut.models import (
    CaptionStyle,
    Clip,
    ClipStatus,
    Layout,
    Project,
    SourceInfo,
    StageName,
    Transcript,
    now_iso,
)
from recut.render.render import render_clip
from recut.store import Store
from recut.text import sentences_from_words, ts

log = get_logger(__name__)

STAGES: tuple[StageName, ...] = ("probe", "audio", "transcribe", "diarize", "clips", "chapters", "posts")
ModelFactory = Callable[[str], ChatModel]


class ReviewError(ValueError):
    """A review action that the guards refuse (e.g. approving a clip with a mid-word cut)."""


class ClipPatch(BaseModel):
    start: float | None = None
    end: float | None = None
    snap: Literal["sentence", "word"] = "word"
    status: ClipStatus | None = None
    force: bool = False
    caption_style: CaptionStyle | None = None
    layout: Layout | None = None
    title: str | None = Field(default=None, max_length=120)
    title_card: bool | None = None
    title_card_text: str | None = Field(default=None, max_length=120)
    caption_overrides: dict[int, str] | None = None
    caption: str | None = Field(default=None, max_length=2200)


class Pipeline:
    def __init__(self, settings: Settings, store: Store | None = None, *, models: ModelFactory | None = None) -> None:
        self.settings = settings
        self.store = store or Store(settings.data_dir)
        self._models = models or (lambda tag: model_for(settings, tag=tag))
        self._levels: dict[str, Levels] = {}

    # ------------------------------------------------------------------ creation
    def create(
        self,
        source: Path,
        *,
        name: str = "",
        license: str = "",
        attribution: str = "",
        source_url: str = "",
        copy: bool = True,
    ) -> Project:
        if not source.exists():
            raise MediaError(f"no such file: {source}")
        project_id = self.store.new_id(name or source.stem)
        folder = self.store.dir(project_id)
        folder.mkdir(parents=True, exist_ok=True)
        if copy:
            target = folder / ("source" + source.suffix.lower())
            shutil.copy2(source, target)
            sidecar = source.with_name(source.name + ".words.json")
            if sidecar.exists():
                shutil.copy2(sidecar, target.with_name(target.name + ".words.json"))
            media = target
        else:
            media = source.resolve()
        project = Project(
            id=project_id,
            name=name or source.stem.replace("-", " ").replace("_", " ").strip(),
            source=SourceInfo(
                path=str(media),
                filename=source.name,
                title=name or source.stem,
                license=license,
                attribution=attribution,
                source_url=source_url,
            ),
        )
        self.audit(project, "created", source=source.name, license=license or "(not set)")
        self.store.save(project)
        return project

    # ------------------------------------------------------------------ processing
    def media(self, project: Project) -> Path:
        return Path(project.source.path)

    def audio_path(self, project: Project) -> Path:
        return self.store.dir(project.id) / "audio.wav"

    def levels(self, project: Project) -> Levels | None:
        if project.id not in self._levels:
            wav = self.audio_path(project)
            if not wav.exists():
                return None
            self._levels[project.id] = Levels.from_samples(load_wav(wav))
        return self._levels[project.id]

    def probe_of(self, project: Project) -> Probe:
        s = project.source
        return Probe(s.duration, s.width, s.height, s.fps, s.has_video, True, s.size_bytes)

    def process(
        self,
        project_id: str,
        *,
        stages: tuple[StageName, ...] = STAGES,
        force: bool = False,
        on_stage: Callable[[Project], None] | None = None,
    ) -> Project:
        project = self.store.load(project_id)
        project.error = ""
        for name in stages:
            stage = project.stage(name)
            if stage.state in ("done", "skipped") and not force:
                continue
            stage.state, stage.message = "running", ""
            self.store.save(project)
            if on_stage:
                on_stage(project)
            started = time.monotonic()
            try:
                message = getattr(self, f"_stage_{name}")(project)
            except Exception as exc:
                stage.state = "failed"
                stage.message = str(exc)[:400]
                stage.seconds = round(time.monotonic() - started, 2)
                project.error = f"{name}: {str(exc)[:300]}"
                self.store.save(project)
                log.error("pipeline.stage_failed", project=project.id, stage=name, error=str(exc)[:300])
                raise
            stage.seconds = round(time.monotonic() - started, 2)
            if message == "skipped":
                stage.state, stage.message = "skipped", ""
            else:
                stage.state, stage.message = "done", message or ""
            self.store.save(project)
            if on_stage:
                on_stage(project)
        return project

    def _stage_probe(self, project: Project) -> str:
        media = self.media(project)
        info = probe(media, ffprobe=self.settings.ffprobe)
        if not info.has_audio:
            raise MediaError("the file has no audio track")
        s = project.source
        s.duration, s.width, s.height, s.fps = info.duration, info.width, info.height, info.fps
        s.has_video, s.size_bytes = info.has_video, info.size_bytes
        s.sha256 = sha256_file(media)
        if info.has_video:
            thumbnail(media, self.store.dir(project.id) / "thumb.jpg", at=min(info.duration * 0.2, 30.0))
        kind = f"{info.width}x{info.height} @ {info.fps:g} fps" if info.has_video else "audio only"
        return f"{ts(info.duration)} · {kind}"

    def _stage_audio(self, project: Project) -> str:
        extract_audio(self.media(project), self.audio_path(project), ffmpeg=self.settings.ffmpeg)
        self._levels.pop(project.id, None)
        return "16 kHz mono"

    def _stage_transcribe(self, project: Project) -> str:
        cache = self.settings.transcript_cache_dir
        key = f"{project.source.sha256}-{self.settings.whisper_model}-{self.settings.whisper_compute_type}.json"
        cached = cache / key if cache is not None and project.source.sha256 and self.settings.asr == "whisper" else None
        if cached is not None and cached.exists():
            transcript = Transcript.model_validate_json(cached.read_text(encoding="utf-8"))
        else:
            transcript = transcribe(
                self.media(project), self.audio_path(project), self.settings, duration=project.source.duration
            )
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(transcript.model_dump_json(), encoding="utf-8")
        project.transcript = transcript
        project.sentences = sentences_from_words(transcript.words)
        rtf = transcript.seconds / max(1.0, project.source.duration)
        return f"{len(transcript.words)} words · {transcript.model} · {rtf:.2f}x real time"

    def _stage_diarize(self, project: Project) -> str:
        transcript = project.transcript
        if transcript is None:
            raise ValueError("no transcript")
        if not self.settings.diarize or transcript.diarization == "reference":
            return "skipped"
        if transcript.diarization not in ("single", "") and len(transcript.speakers) > 1:
            return f"{len(transcript.speakers)} speakers (from the sidecar)"
        samples = load_wav(self.audio_path(project))
        result = diarization.diarize(transcript, samples, max_speakers=self.settings.max_speakers)
        project.transcript = diarization.apply(transcript, result)
        project.sentences = sentences_from_words(project.transcript.words)
        return result.note

    def _stage_clips(self, project: Project) -> str:
        transcript = self._transcript(project)
        model = self._models("clips")
        report = suggest(
            project.sentences,
            transcript.words,
            self.levels(project),
            model,
            self.settings,
            title=project.name,
            media_end=project.source.duration,
        )
        manual = [c for c in project.clips if c.source == "manual" or c.status != "suggested"]
        project.clips = manual + report.clips
        self._rerank(project)
        project.llm = report.model
        dropped = f", {len(report.rejected)} dropped" if report.rejected else ""
        return f"{len(report.clips)} clips from {report.candidates} candidates{dropped}"

    def _stage_chapters(self, project: Project) -> str:
        model = self._models("chapters")
        chapters, _ = make_chapters(
            project.sentences, model, self.settings, duration=project.source.duration, title=project.name
        )
        project.chapters = chapters
        project.chapters_text = format_youtube(chapters)
        problems = youtube_ready(chapters, project.source.duration)
        return f"{len(chapters)} chapters" + (f" ({'; '.join(problems)})" if problems else "")

    def _stage_posts(self, project: Project) -> str:
        model = self._models("posts")
        outputs, stats = generate(project.sentences, project.chapters, project.source, model, self.settings)
        project.outputs = outputs
        flagged = len(outputs.issues) + sum(len(p.issues) for p in outputs.social)
        errors = f", errors: {'; '.join(stats['errors'])}" if stats["errors"] else ""
        return f"{len(outputs.social)} social posts, {flagged} grounding flags{errors}"

    def _transcript(self, project: Project) -> Transcript:
        if project.transcript is None:
            raise ValueError("the project has no transcript yet")
        return project.transcript

    def _rerank(self, project: Project) -> None:
        ordered = sorted(project.clips, key=lambda c: c.score, reverse=True)
        for rank, clip in enumerate(ordered, start=1):
            clip.rank = rank
        project.clips = ordered

    # ------------------------------------------------------------------ review
    def audit(self, project: Project, action: str, **details: Any) -> None:
        project.audit.append({"at": now_iso(), "action": action, **details})
        project.audit = project.audit[-500:]

    def update_clip(self, project_id: str, clip_id: str, patch: ClipPatch) -> Project:
        with self.store.lock(project_id):
            project = self.store.load(project_id)
            clip = project.clip(clip_id)
            transcript = self._transcript(project)
            before = {"start": clip.start, "end": clip.end, "status": clip.status}
            changed_cut = patch.start is not None or patch.end is not None
            if changed_cut:
                start = patch.start if patch.start is not None else clip.start
                end = patch.end if patch.end is not None else clip.end
                if end - start < 3.0:
                    raise ReviewError("a clip must be at least 3 seconds long")
                result = snap_times(
                    start,
                    end,
                    project.sentences,
                    transcript.words,
                    self.levels(project),
                    min_s=0.0 if patch.snap == "word" else self.settings.clip_min_seconds,
                    max_s=10_000.0 if patch.snap == "word" else self.settings.clip_max_seconds,
                    pad_in=self.settings.pad_in_seconds,
                    pad_out=self.settings.pad_out_seconds,
                    to_sentences=patch.snap == "sentence",
                    media_end=project.source.duration,
                )
                clip.start, clip.end = result.start, result.end
                clip.first_sentence, clip.last_sentence = result.first_sentence, result.last_sentence
                clip.cut_in, clip.cut_out = result.cut_in, result.cut_out
                clip.adjustments = result.adjustments
                clip.edited = True
                clip.hook = project.sentences[clip.first_sentence].text
            if patch.caption_style is not None:
                clip.caption_style = patch.caption_style
            if patch.layout is not None:
                clip.layout = patch.layout
                clip.crop = None
            if patch.title is not None:
                clip.title = patch.title.strip() or clip.title
            if patch.title_card is not None:
                clip.title_card = patch.title_card
            if patch.title_card_text is not None:
                clip.title_card_text = patch.title_card_text.strip()
            if patch.caption is not None:
                clip.post.caption = patch.caption.strip()
            if patch.caption_overrides is not None:
                words = transcript.words
                for index, text in patch.caption_overrides.items():
                    if not 0 <= index < len(words):
                        raise ReviewError(f"word {index} does not exist")
                    w = words[index]
                    if w.end <= clip.start or w.start >= clip.end:
                        raise ReviewError(f"word {index} is not inside this clip")
                    if len(text) > 40:
                        raise ReviewError("a caption word is at most 40 characters")
                    if text == w.text:
                        clip.caption_overrides.pop(index, None)
                    else:
                        clip.caption_overrides[index] = text
                clip.edited = True
            clip.issues = lint_clip(
                clip.first_sentence,
                clip.last_sentence,
                project.sentences,
                transcript.words,
                start=clip.start,
                end=clip.end,
                min_s=self.settings.clip_min_seconds,
                max_s=self.settings.clip_max_seconds,
                cut_in=clip.cut_in,
                cut_out=clip.cut_out,
            )
            clip.score = rank_score(clip.scores, penalty(clip.issues)) if clip.source != "manual" else clip.score
            if changed_cut or patch.caption_overrides is not None or patch.caption_style or patch.layout:
                clip.render = None
            if patch.title_card is not None or patch.title_card_text is not None:
                clip.render = None
            if patch.status is not None:
                if patch.status == "approved":
                    errors = [i for i in clip.issues if i.severity == "error"]
                    if errors and not patch.force:
                        raise ReviewError(
                            "fix these before approving (or approve with force): "
                            + "; ".join(i.message for i in errors)
                        )
                clip.status = patch.status
            clip.history.append({"at": now_iso(), "before": before, "patch": patch.model_dump(exclude_none=True)})
            clip.history = clip.history[-50:]
            self.audit(project, "clip_updated", clip=clip.id, fields=sorted(patch.model_dump(exclude_none=True)))
            self.store.save(project)
            return project

    def add_clip(self, project_id: str, start: float, end: float, *, title: str = "") -> Project:
        with self.store.lock(project_id):
            project = self.store.load(project_id)
            transcript = self._transcript(project)
            result = snap_times(
                start,
                end,
                project.sentences,
                transcript.words,
                self.levels(project),
                min_s=self.settings.clip_min_seconds,
                max_s=self.settings.clip_max_seconds,
                pad_in=self.settings.pad_in_seconds,
                pad_out=self.settings.pad_out_seconds,
                media_end=project.source.duration,
            )
            issues = lint_clip(
                result.first_sentence,
                result.last_sentence,
                project.sentences,
                transcript.words,
                start=result.start,
                end=result.end,
                min_s=self.settings.clip_min_seconds,
                max_s=self.settings.clip_max_seconds,
                cut_in=result.cut_in,
                cut_out=result.cut_out,
            )
            opening = project.sentences[result.first_sentence].text
            clip = Clip(
                id=f"clip_m{len(project.clips) + 1:02d}{int(time.time()) % 1000:03d}",
                start=result.start,
                end=result.end,
                first_sentence=result.first_sentence,
                last_sentence=result.last_sentence,
                title=title or opening[:60],
                hook=opening,
                reason="added by the reviewer",
                cut_in=result.cut_in,
                cut_out=result.cut_out,
                issues=issues,
                adjustments=result.adjustments,
                source="manual",
                score=max(0.0, 50.0 - penalty(issues)),
            )
            clip.title_card_text = clip.title
            project.clips.append(clip)
            self._rerank(project)
            self.audit(project, "clip_added", clip=clip.id, start=clip.start, end=clip.end)
            self.store.save(project)
            return project

    def render_path(self, project: Project, clip: Clip) -> Path:
        return self.store.dir(project.id) / "renders" / f"{clip.id}.mp4"

    def render(self, project_id: str, clip_id: str) -> Project:
        project = self.store.load(project_id)
        clip = project.clip(clip_id)
        transcript = self._transcript(project)
        info = render_clip(
            self.media(project),
            self.probe_of(project),
            clip,
            transcript.words,
            self.render_path(project, clip),
            self.settings,
        )
        with self.store.lock(project_id):
            fresh = self.store.load(project_id)
            target = fresh.clip(clip_id)
            target.render = info
            target.crop = clip.crop
            self.audit(fresh, "rendered", clip=clip_id, seconds=info.seconds, layout=info.layout)
            self.store.save(fresh)
            return fresh

    def update_outputs(self, project_id: str, patch: dict[str, Any]) -> Project:
        allowed = {"seo_title", "seo_description", "tags", "show_notes", "blog_post", "approved", "chapters_text"}
        unknown = set(patch) - allowed - {"social"}
        if unknown:
            raise ReviewError(f"unknown fields: {', '.join(sorted(unknown))}")
        with self.store.lock(project_id):
            project = self.store.load(project_id)
            for key, value in patch.items():
                if key == "chapters_text":
                    project.chapters_text = str(value)
                elif key == "social":
                    for item in value or []:
                        for post in project.outputs.social:
                            if post.platform == item.get("platform"):
                                if "text" in item:
                                    from recut.content import limits

                                    post.text = str(item["text"])
                                    post.chars = limits.count(post.text, limits.PLATFORMS[post.platform].counting)
                                    post.within_limit = post.chars <= post.limit
                                if "approved" in item:
                                    if item["approved"] and not post.within_limit:
                                        raise ReviewError(f"{post.platform}: over the {post.limit}-character limit")
                                    post.approved = bool(item["approved"])
                else:
                    setattr(project.outputs, key, value)
            self.audit(project, "outputs_updated", fields=sorted(patch))
            self.store.save(project)
            return project

    def export_bundle(self, project_id: str) -> bytes:
        """A zip of the approved clips (rendered), captions as SRT, chapters and the approved texts. Refuses when
        nothing is approved or when the source has no licence/provenance recorded."""
        project = self.store.load(project_id)
        if not project.source.license.strip():
            raise ReviewError("record the source licence (and attribution if required) before exporting")
        approved = [c for c in project.clips if c.status == "approved"]
        if not approved and not project.outputs.approved:
            raise ReviewError("nothing is approved yet")
        missing = [c.id for c in approved if c.render is None or not Path(c.render.path).exists()]
        if missing:
            raise ReviewError(f"render the approved clips first: {', '.join(missing)}")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest: dict[str, Any] = {
                "project": project.name,
                "source": project.source.model_dump(exclude={"path"}),
                "exported_at": now_iso(),
                "clips": [],
            }
            for clip in approved:
                assert clip.render is not None
                stem = f"{clip.rank:02d}-{_slug(clip.title)}"
                zf.write(clip.render.path, f"clips/{stem}.mp4")
                zf.writestr(f"clips/{stem}.srt", clip_srt(project, clip))
                caption = clip.post.caption + ("\n\n" + " ".join(clip.post.hashtags) if clip.post.hashtags else "")
                zf.writestr(f"clips/{stem}.txt", f"{clip.title}\n\n{caption.strip()}\n")
                manifest["clips"].append(
                    {"file": f"clips/{stem}.mp4", "start": clip.start, "end": clip.end, "title": clip.title}
                )
            if project.outputs.approved:
                o = project.outputs
                zf.writestr("youtube/title.txt", o.seo_title + "\n")
                zf.writestr("youtube/description.txt", o.seo_description + "\n")
                zf.writestr("youtube/tags.txt", ", ".join(o.tags) + "\n")
                zf.writestr("youtube/chapters.txt", project.chapters_text + "\n")
                zf.writestr("show-notes.md", o.show_notes)
                zf.writestr("blog-post.md", o.blog_post)
                for post in o.social:
                    if post.approved:
                        zf.writestr(f"social/{post.platform}.txt", post.text + "\n")
            zf.writestr("manifest.json", json.dumps(manifest, indent=1))
        self.audit(project, "exported", clips=len(approved), texts=project.outputs.approved)
        with self.store.lock(project_id):
            fresh = self.store.load(project_id)
            fresh.audit = project.audit
            self.store.save(fresh)
        return buf.getvalue()


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "clip"


def clip_srt(project: Project, clip: Clip) -> str:
    """SRT of the clip's captions (times relative to the clip), with the reviewer's corrections."""
    from recut.render.captions import STYLES, clip_words, group_words

    if project.transcript is None:
        return ""
    words = clip_words(project.transcript.words, clip.start, clip.end, clip.caption_overrides)
    groups = group_words(words, STYLES["minimal"])

    def stamp(t: float) -> str:
        ms = int(round(t * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, group in enumerate(groups, start=1):
        blocks.append(f"{i}\n{stamp(group[0].start)} --> {stamp(group[-1].end)}\n{' '.join(w.text for w in group)}\n")
    return "\n".join(blocks)
