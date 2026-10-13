"""A small in-process job runner: one worker for processing (CPU-heavy transcription), two for renders."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from recut.logging_config import get_logger

log = get_logger(__name__)

JobKind = Literal["process", "render"]
JobState = Literal["queued", "running", "done", "failed"]


@dataclass
class Job:
    id: str
    kind: JobKind
    project_id: str
    clip_id: str = ""
    state: JobState = "queued"
    error: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["seconds"] = round((self.finished or time.time()) - self.started, 1) if self.started else 0.0
        return data


class JobRunner:
    def __init__(self) -> None:
        self._process = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recut-process")
        self._render = ThreadPoolExecutor(max_workers=2, thread_name_prefix="recut-render")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: JobKind, project_id: str, fn: Callable[[], object], *, clip_id: str = "") -> Job:
        with self._lock:
            for job in self._jobs.values():
                if (
                    job.kind == kind
                    and job.project_id == project_id
                    and job.clip_id == clip_id
                    and job.state in ("queued", "running")
                ):
                    return job  # the same work is already queued
            job = Job(id="job_" + uuid.uuid4().hex[:10], kind=kind, project_id=project_id, clip_id=clip_id)
            self._jobs[job.id] = job

        def runner() -> None:
            job.state, job.started = "running", time.time()
            try:
                fn()
                job.state = "done"
            except Exception as exc:
                job.state, job.error = "failed", str(exc)[:500]
                log.error(
                    "job.failed", job=job.id, kind=kind, error=str(exc)[:300], trace=traceback.format_exc()[-800:]
                )
            finally:
                job.finished = time.time()

        (self._process if kind == "process" else self._render).submit(runner)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def for_project(self, project_id: str) -> list[Job]:
        return sorted((j for j in self._jobs.values() if j.project_id == project_id), key=lambda j: j.created)

    def shutdown(self) -> None:
        self._process.shutdown(wait=False, cancel_futures=True)
        self._render.shutdown(wait=False, cancel_futures=True)
