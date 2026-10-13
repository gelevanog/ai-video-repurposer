from __future__ import annotations

from pathlib import Path

import pytest

from recut.models import Project, SourceInfo
from recut.store import ProjectNotFoundError, Store


def _project(pid: str = "demo-abc123") -> Project:
    return Project(id=pid, name="Demo", source=SourceInfo(path="/x.mp4", filename="x.mp4"))


def test_save_load_and_list(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.save(_project())
    store.save(_project("other-def456"))
    assert store.load("demo-abc123").name == "Demo"
    assert {p.id for p in store.list()} == {"demo-abc123", "other-def456"}
    assert store.exists("demo-abc123") and not store.exists("nope-000000")


def test_a_corrupt_project_falls_back_to_the_backup(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.save(_project())
    project = store.load("demo-abc123")
    project.name = "Second save"
    store.save(project)
    (tmp_path / "demo-abc123" / "project.json").write_text("\x00\x00\x00")  # a zero-filled file after a crash
    assert store.load("demo-abc123").name == "Demo"


def test_ids_are_validated(tmp_path: Path) -> None:
    store = Store(tmp_path)
    for bad in ("../etc", "A", "x/y", ""):
        with pytest.raises(ProjectNotFoundError):
            store.dir(bad)
    with pytest.raises(ProjectNotFoundError):
        store.load("missing-123456")


def test_new_ids_are_slugs() -> None:
    pid = Store.new_id("My Great Video!!")
    assert pid.startswith("my-great-video-") and len(pid) <= 64


def test_delete(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.save(_project())
    store.delete("demo-abc123")
    assert store.list() == []


def test_project_status_and_lookup() -> None:
    project = _project()
    assert project.status == "queued"
    project.stage("probe").state = "running"
    assert project.status == "processing"
    for stage in project.stages:
        stage.state = "done"
    assert project.status == "ready"
    project.stage("posts").state = "failed"
    assert project.status == "failed"
    with pytest.raises(KeyError):
        project.clip("nope")
