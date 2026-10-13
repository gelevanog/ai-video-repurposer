"""Projects on disk: one folder per project with project.json, the 16 kHz audio, thumbnails and renders.

project.json is written atomically (temp file, fsync, rename) and the previous version is kept as
project.json.bak, so a crash mid-write never leaves a half-written project."""

from __future__ import annotations

import os
import re
import shutil
import threading
import uuid
from pathlib import Path

from recut.models import Project, now_iso

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


class ProjectNotFoundError(KeyError):
    pass


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def lock(self, project_id: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(project_id, threading.RLock())

    @staticmethod
    def new_id(name: str = "") -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32].strip("-")
        return f"{slug or 'project'}-{uuid.uuid4().hex[:6]}"

    def dir(self, project_id: str) -> Path:
        if not _ID.match(project_id):
            raise ProjectNotFoundError(project_id)
        return self.root / project_id

    def path(self, project_id: str) -> Path:
        return self.dir(project_id) / "project.json"

    def exists(self, project_id: str) -> bool:
        try:
            return self.path(project_id).exists()
        except ProjectNotFoundError:
            return False

    def load(self, project_id: str) -> Project:
        path = self.path(project_id)
        if not path.exists():
            raise ProjectNotFoundError(project_id)
        try:
            return Project.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            backup = path.with_name("project.json.bak")
            if backup.exists():
                return Project.model_validate_json(backup.read_text(encoding="utf-8"))
            raise

    def save(self, project: Project) -> None:
        with self.lock(project.id):
            project.updated_at = now_iso()
            folder = self.dir(project.id)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "project.json"
            tmp = folder / f".project.{uuid.uuid4().hex[:6]}.tmp"
            data = project.model_dump_json(indent=1)
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                shutil.copy2(path, folder / "project.json.bak")
            tmp.replace(path)

    def list(self) -> list[Project]:
        out = []
        for child in sorted(self.root.iterdir()) if self.root.exists() else []:
            if (child / "project.json").exists():
                try:
                    out.append(self.load(child.name))
                except (ValueError, ProjectNotFoundError):
                    continue
        out.sort(key=lambda p: p.created_at, reverse=True)
        return out

    def delete(self, project_id: str) -> None:
        folder = self.dir(project_id)
        if folder.exists():
            shutil.rmtree(folder)
