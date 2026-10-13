"""HTTP API for the review UI: projects, processing jobs, the clip editor, renders, outputs and exports.

Safety in the API, not only in the UI: a clip can only be downloaded once a person has approved it, an export
needs the source licence recorded, approving a clip with a mid-word cut needs an explicit force, and a local path
can only be imported from the configured folders."""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from recut import __version__
from recut.config import REPO_ROOT, Settings, get_settings
from recut.content import limits
from recut.logging_config import configure_logging, get_logger
from recut.media.ffmpeg import MediaError
from recut.models import Project
from recut.pipeline import ClipPatch, Pipeline, ReviewError, clip_srt
from recut.render.captions import STYLE_LABELS
from recut.store import ProjectNotFoundError

from .jobs import JobRunner

log = get_logger(__name__)

MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}
ALLOWED_SUFFIXES = set(MEDIA_TYPES)


class FromPath(BaseModel):
    path: str
    name: str = ""
    license: str = ""
    attribution: str = ""
    source_url: str = ""


class NewClip(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    title: str = Field(default="", max_length=120)


class ProcessRequest(BaseModel):
    stages: list[str] = Field(default_factory=list)
    force: bool = False


class SourcePatch(BaseModel):
    license: str | None = Field(default=None, max_length=200)
    attribution: str | None = Field(default=None, max_length=400)
    source_url: str | None = Field(default=None, max_length=500)
    name: str | None = Field(default=None, max_length=120)


def summary(project: Project) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "status": project.status,
        "created_at": project.created_at,
        "duration": project.source.duration,
        "has_video": project.source.has_video,
        "license": project.source.license,
        "clips": len(project.clips),
        "approved": sum(1 for c in project.clips if c.status == "approved"),
        "rendered": sum(1 for c in project.clips if c.render is not None),
        "speakers": len(project.transcript.speakers) if project.transcript else 0,
        "chapters": len(project.chapters),
        "llm": project.llm,
        "error": project.error,
        "stages": [s.model_dump() for s in project.stages],
    }


def create_app(settings: Settings | None = None, *, pipeline: Pipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    pipe = pipeline or Pipeline(settings)
    jobs = JobRunner()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.seed_demo and not pipe.store.list():
            _seed_demo(pipe, settings, jobs)
        yield
        jobs.shutdown()

    app = FastAPI(title="Recut API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )
    app.state.pipeline = pipe
    app.state.jobs = jobs

    def load(project_id: str) -> Project:
        try:
            return pipe.store.load(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(404, f"no project {project_id}") from exc

    def enqueue_process(project_id: str, stages: tuple[Any, ...] | None = None, force: bool = False) -> dict[str, Any]:
        def work() -> None:
            if stages:
                pipe.process(project_id, stages=stages, force=force)
            else:
                pipe.process(project_id, force=force)

        return jobs.submit("process", project_id, work).as_dict()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "version": __version__,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.model_name,
            "free_only": settings.free_only,
            "asr": settings.asr,
            "whisper_model": settings.whisper_model,
            "clip_min_seconds": settings.clip_min_seconds,
            "clip_max_seconds": settings.clip_max_seconds,
            "styles": STYLE_LABELS,
            "platforms": {
                k: {"name": p.name, "limit": p.limit, "target": p.target} for k, p in limits.PLATFORMS.items()
            },
            "youtube": {
                "title": limits.YOUTUBE_TITLE,
                "title_target": limits.YOUTUBE_TITLE_TARGET,
                "description": limits.YOUTUBE_DESCRIPTION,
            },
            "samples": _importable(settings),
        }

    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        return [summary(p) for p in pipe.store.list()]

    @app.post("/api/projects", status_code=201)
    async def upload(
        file: Annotated[UploadFile, File()],
        name: Annotated[str, Form()] = "",
        license: Annotated[str, Form()] = "",
        attribution: Annotated[str, Form()] = "",
        source_url: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        suffix = Path(file.filename or "upload.mp4").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                415, f"unsupported file type {suffix or '(none)'}; use {', '.join(sorted(ALLOWED_SUFFIXES))}"
            )
        limit = settings.max_upload_mb * 1024 * 1024
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / ("upload" + suffix)
            size = 0
            with tmp.open("wb") as handle:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
                    handle.write(chunk)
            try:
                project = pipe.create(
                    tmp,
                    name=name or Path(file.filename or "upload").stem,
                    license=license,
                    attribution=attribution,
                    source_url=source_url,
                )
            except MediaError as exc:
                raise HTTPException(400, str(exc)) from exc
        project.source.filename = file.filename or project.source.filename
        pipe.store.save(project)
        job = enqueue_process(project.id)
        return {"project": summary(project), "job": job}

    @app.post("/api/projects/from-path", status_code=201)
    def from_path(body: FromPath) -> dict[str, Any]:
        path = Path(body.path).expanduser().resolve()
        allowed = [d.resolve() for d in _allowed_dirs(settings)]
        if not any(path.is_relative_to(d) for d in allowed):
            raise HTTPException(403, "importing from this folder is not allowed (RECUT_ALLOWED_SOURCE_DIRS)")
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(404, "no such media file")
        project = pipe.create(
            path,
            name=body.name or path.stem,
            license=body.license,
            attribution=body.attribution,
            source_url=body.source_url,
            copy=path.stat().st_size < 200 * 1024 * 1024,
        )
        return {"project": summary(project), "job": enqueue_process(project.id)}

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        project = load(project_id)
        data = project.model_dump(mode="json")
        data["status"] = project.status
        data["jobs"] = [j.as_dict() for j in jobs.for_project(project_id)]
        return data

    @app.patch("/api/projects/{project_id}")
    def patch_project(project_id: str, body: SourcePatch) -> dict[str, Any]:
        with pipe.store.lock(project_id):
            project = load(project_id)
            for key, value in body.model_dump(exclude_none=True).items():
                if key == "name":
                    project.name = value
                else:
                    setattr(project.source, key, value)
            pipe.audit(project, "source_updated", fields=sorted(body.model_dump(exclude_none=True)))
            pipe.store.save(project)
        return summary(project)

    @app.delete("/api/projects/{project_id}", status_code=204)
    def delete_project(project_id: str) -> Response:
        load(project_id)
        pipe.store.delete(project_id)
        return Response(status_code=204)

    @app.post("/api/projects/{project_id}/process")
    def reprocess(project_id: str, body: ProcessRequest | None = None) -> dict[str, Any]:
        load(project_id)
        body = body or ProcessRequest()
        valid = {"probe", "audio", "transcribe", "diarize", "clips", "chapters", "posts"}
        bad = [s for s in body.stages if s not in valid]
        if bad:
            raise HTTPException(422, f"unknown stages: {', '.join(bad)}")
        return enqueue_process(project_id, tuple(body.stages) or None, force=body.force)

    @app.get("/api/projects/{project_id}/media")
    def media(project_id: str) -> FileResponse:
        project = load(project_id)
        path = Path(project.source.path)
        if not path.exists():
            raise HTTPException(404, "the source file is gone")
        return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"))

    @app.get("/api/projects/{project_id}/thumb")
    def thumb(project_id: str) -> FileResponse:
        path = pipe.store.dir(load(project_id).id) / "thumb.jpg"
        if not path.exists():
            raise HTTPException(404, "no thumbnail")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/projects/{project_id}/waveform")
    def waveform(project_id: str, per_second: Annotated[int, Query(ge=1, le=20)] = 4) -> dict[str, Any]:
        project = load(project_id)
        levels = pipe.levels(project)
        if levels is None:
            return {"per_second": per_second, "values": []}
        step = max(1, round(1 / (levels.frame * per_second)))
        n = len(levels.db) // step
        db = levels.db[: n * step].reshape(n, step).max(axis=1) if n else np.zeros(0)
        floor = float(np.percentile(db, 5)) if n else -60.0
        top = float(np.percentile(db, 99)) if n else 0.0
        values = np.clip((db - floor) / max(1e-6, top - floor), 0, 1)
        return {"per_second": per_second, "values": [round(float(v), 3) for v in values]}

    @app.patch("/api/projects/{project_id}/clips/{clip_id}")
    def patch_clip(project_id: str, clip_id: str, patch: ClipPatch) -> dict[str, Any]:
        load(project_id)
        try:
            project = pipe.update_clip(project_id, clip_id, patch)
        except KeyError as exc:
            raise HTTPException(404, f"no clip {clip_id}") from exc
        except ReviewError as exc:
            raise HTTPException(409, str(exc)) from exc
        return project.clip(clip_id).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/clips", status_code=201)
    def add_clip(project_id: str, body: NewClip) -> dict[str, Any]:
        load(project_id)
        if body.end <= body.start:
            raise HTTPException(422, "end must be after start")
        try:
            project = pipe.add_clip(project_id, body.start, body.end, title=body.title)
        except (ReviewError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        newest = max(project.clips, key=lambda c: c.id if c.source == "manual" else "")
        return newest.model_dump(mode="json")

    @app.post("/api/projects/{project_id}/clips/{clip_id}/render")
    def render(project_id: str, clip_id: str) -> dict[str, Any]:
        project = load(project_id)
        try:
            project.clip(clip_id)
        except KeyError as exc:
            raise HTTPException(404, f"no clip {clip_id}") from exc
        return jobs.submit("render", project_id, lambda: pipe.render(project_id, clip_id), clip_id=clip_id).as_dict()

    @app.get("/api/projects/{project_id}/clips/{clip_id}/framing")
    def clip_framing(project_id: str, clip_id: str) -> dict[str, Any]:
        project = load(project_id)
        try:
            clip = project.clip(clip_id)
        except KeyError as exc:
            raise HTTPException(404, f"no clip {clip_id}") from exc
        if clip.crop is None:
            try:
                clip = pipe.framing(project_id, clip_id).clip(clip_id)
            except MediaError as exc:
                raise HTTPException(500, str(exc)) from exc
        assert clip.crop is not None
        return clip.crop.model_dump(mode="json")

    @app.get("/api/projects/{project_id}/clips/{clip_id}/video")
    def clip_video(project_id: str, clip_id: str, download: bool = False) -> FileResponse:
        project = load(project_id)
        try:
            clip = project.clip(clip_id)
        except KeyError as exc:
            raise HTTPException(404, f"no clip {clip_id}") from exc
        if clip.render is None or not Path(clip.render.path).exists():
            raise HTTPException(404, "not rendered yet")
        if download and clip.status != "approved":
            raise HTTPException(403, "approve the clip before downloading it")
        filename = f"{clip.rank:02d}-{clip.id}.mp4" if download else None
        return FileResponse(clip.render.path, media_type="video/mp4", filename=filename)

    @app.get("/api/projects/{project_id}/clips/{clip_id}/srt", response_class=PlainTextResponse)
    def clip_captions(project_id: str, clip_id: str) -> str:
        project = load(project_id)
        try:
            return clip_srt(project, project.clip(clip_id))
        except KeyError as exc:
            raise HTTPException(404, f"no clip {clip_id}") from exc

    @app.patch("/api/projects/{project_id}/outputs")
    def patch_outputs(project_id: str, body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        load(project_id)
        try:
            project = pipe.update_outputs(project_id, body)
        except ReviewError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"outputs": project.outputs.model_dump(mode="json"), "chapters_text": project.chapters_text}

    @app.get("/api/projects/{project_id}/export")
    def export(project_id: str) -> Response:
        load(project_id)
        try:
            data = pipe.export_bundle(project_id)
        except ReviewError as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{project_id}-recut.zip"'},
        )

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        found = jobs.get(job_id)
        if found is None:
            raise HTTPException(404, "no such job")
        return found.as_dict()

    @app.get("/api/eval")
    def evaluation() -> dict[str, Any]:
        path = REPO_ROOT / "results" / "summary.json"
        if not path.exists():
            return {}
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    return app


def _allowed_dirs(settings: Settings) -> list[Path]:
    return [
        settings.samples_dir,
        REPO_ROOT / "data" / "testset" / "media",
        *(Path(d) for d in settings.allowed_source_dirs),
    ]


def _importable(settings: Settings) -> list[dict[str, str]]:
    """Media in the allowed folders, with the licence and credit from the test-set manifest when it is listed."""
    known: dict[str, dict[str, str]] = {}
    manifest = REPO_ROOT / "data" / "testset" / "manifest.json"
    if manifest.exists():
        for item in json.loads(manifest.read_text(encoding="utf-8")).get("items", []):
            known[f"{item['id']}.mp4"] = item
    out: list[dict[str, str]] = []
    for folder in _allowed_dirs(settings):
        if not folder.exists():
            continue
        for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES):
            item = known.get(path.name, {})
            name = item.get("title") or path.stem.replace("-", " ")
            if any(o["name"] == name for o in out):
                continue
            out.append(
                {
                    "name": item.get("title") or path.stem.replace("-", " "),
                    "path": str(path),
                    "license": item.get("license") or "CC0 1.0 (synthetic sample)",
                    "attribution": item.get("attribution") or "",
                }
            )
    return out


def _seed_demo(pipe: Pipeline, settings: Settings, jobs: JobRunner) -> None:
    sample = settings.samples_dir / "creator-margin-ep12.mp4"
    if not sample.exists():
        return
    project = pipe.create(
        sample,
        name="The Creator Margin, episode 12",
        license="CC0 1.0 (synthetic sample)",
        attribution="",
    )
    log.info("api.seeded_demo", project=project.id)
    jobs.submit("process", project.id, lambda: pipe.process(project.id))


def create_default_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    return create_app(settings)


__all__ = ["create_app", "create_default_app"]
