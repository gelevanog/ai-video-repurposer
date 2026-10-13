"""The HTTP API with FastAPI's TestClient: jobs run in the background thread pool, so tests poll the job."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from recut.api.app import create_app
from recut.config import Settings
from tests.conftest import needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture()
def client(settings: Settings, sample: Path) -> TestClient:
    s = settings.model_copy(update={"allowed_source_dirs": [str(sample.parent)]})
    with TestClient(create_app(s)) as c:
        yield c  # type: ignore[misc]


def _wait(client: TestClient, job_id: str, timeout: float = 120) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "failed"):
            return job  # type: ignore[no-any-return]
        time.sleep(0.2)
    raise AssertionError("job timed out")


@pytest.fixture()
def project(client: TestClient, sample: Path) -> dict[str, Any]:
    created = client.post("/api/projects/from-path", json={"path": str(sample), "license": "CC0 1.0"})
    assert created.status_code == 201
    job = _wait(client, created.json()["job"]["id"])
    assert job["state"] == "done", job
    return client.get(f"/api/projects/{created.json()['project']['id']}").json()  # type: ignore[no-any-return]


def test_health_and_config(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    config = client.get("/api/config").json()
    assert config["llm_provider"] == "fake" and config["free_only"] is True
    assert set(config["styles"]) == {"bold", "boxed", "karaoke", "pop", "minimal"}
    assert config["platforms"]["x"]["limit"] == 280


def test_import_processes_the_project(project: dict[str, Any], client: TestClient) -> None:
    assert project["status"] == "ready" and project["clips"] and project["chapters"]
    listed = client.get("/api/projects").json()
    assert listed[0]["id"] == project["id"] and listed[0]["clips"] == len(project["clips"])


def test_imports_outside_the_allowed_folders_are_refused(client: TestClient, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(b"x")
    assert client.post("/api/projects/from-path", json={"path": str(outside)}).status_code == 403
    assert client.post("/api/projects/from-path", json={"path": "/etc/passwd"}).status_code == 403


def test_upload_rejects_unknown_types(client: TestClient) -> None:
    response = client.post("/api/projects", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_upload_creates_a_project(client: TestClient, sample: Path) -> None:
    with sample.open("rb") as handle:
        response = client.post(
            "/api/projects", files={"file": ("talk.mp4", handle, "video/mp4")}, data={"license": "CC0"}
        )
    assert response.status_code == 201
    assert response.json()["project"]["license"] == "CC0"
    _wait(client, response.json()["job"]["id"])


def test_media_supports_range_requests(project: dict[str, Any], client: TestClient) -> None:
    response = client.get(f"/api/projects/{project['id']}/media", headers={"Range": "bytes=0-99"})
    assert response.status_code == 206 and len(response.content) == 100
    assert client.get(f"/api/projects/{project['id']}/thumb").headers["content-type"] == "image/jpeg"


def test_waveform(project: dict[str, Any], client: TestClient) -> None:
    wave = client.get(f"/api/projects/{project['id']}/waveform?per_second=4").json()
    assert wave["per_second"] == 4 and abs(len(wave["values"]) - project["source"]["duration"] * 4) < 8
    assert min(wave["values"]) >= 0 and max(wave["values"]) <= 1


def test_clip_edit_render_approve_download_export(project: dict[str, Any], client: TestClient) -> None:
    pid, clip = project["id"], project["clips"][0]
    base = f"/api/projects/{pid}/clips/{clip['id']}"
    patched = client.patch(base, json={"caption_style": "boxed", "title_card_text": "My title"}).json()
    assert patched["caption_style"] == "boxed" and patched["title_card_text"] == "My title"
    assert client.get(f"{base}/video").status_code == 404
    job = client.post(f"{base}/render").json()
    assert _wait(client, job["id"])["state"] == "done"
    assert client.get(f"{base}/video").status_code == 200
    assert client.get(f"{base}/video?download=1").status_code == 403  # not approved yet
    assert client.get(f"/api/projects/{pid}/export").status_code == 409
    assert client.patch(base, json={"status": "approved"}).json()["status"] == "approved"
    download = client.get(f"{base}/video?download=1")
    assert download.status_code == 200 and "attachment" in download.headers["content-disposition"]
    export = client.get(f"/api/projects/{pid}/export")
    assert export.status_code == 200 and export.headers["content-type"] == "application/zip"
    assert client.get(f"{base}/srt").text.startswith("1\n")


def test_clip_errors(project: dict[str, Any], client: TestClient) -> None:
    pid = project["id"]
    assert client.patch(f"/api/projects/{pid}/clips/nope", json={}).status_code == 404
    clip = project["clips"][0]
    short = client.patch(f"/api/projects/{pid}/clips/{clip['id']}", json={"end": clip["start"] + 1})
    assert short.status_code == 409
    assert (
        client.patch(f"/api/projects/{pid}/clips/{clip['id']}", json={"caption_style": "comic-sans"}).status_code == 422
    )
    assert client.get("/api/projects/missing-000000").status_code == 404


def test_manual_clip_and_outputs(project: dict[str, Any], client: TestClient) -> None:
    pid = project["id"]
    created = client.post(f"/api/projects/{pid}/clips", json={"start": 20, "end": 50, "title": "Mine"})
    assert created.status_code == 201 and created.json()["source"] == "manual"
    outputs = client.patch(f"/api/projects/{pid}/outputs", json={"seo_title": "New title", "approved": True}).json()
    assert outputs["outputs"]["seo_title"] == "New title" and outputs["outputs"]["approved"] is True
    assert client.patch(f"/api/projects/{pid}/outputs", json={"bogus": 1}).status_code == 409


def test_source_licence_can_be_recorded(project: dict[str, Any], client: TestClient) -> None:
    updated = client.patch(
        f"/api/projects/{project['id']}", json={"license": "CC BY 4.0", "attribution": "Jane Doe"}
    ).json()
    assert updated["license"] == "CC BY 4.0"
    assert client.get(f"/api/projects/{project['id']}").json()["source"]["attribution"] == "Jane Doe"


def test_reprocess_validates_stages(project: dict[str, Any], client: TestClient) -> None:
    pid = project["id"]
    assert client.post(f"/api/projects/{pid}/process", json={"stages": ["nope"]}).status_code == 422
    job = client.post(f"/api/projects/{pid}/process", json={"stages": ["chapters"], "force": True}).json()
    assert _wait(client, job["id"])["state"] == "done"


def test_delete_project(project: dict[str, Any], client: TestClient) -> None:
    assert client.delete(f"/api/projects/{project['id']}").status_code == 204
    assert client.get(f"/api/projects/{project['id']}").status_code == 404


def test_demo_project_is_seeded_on_an_empty_store(settings: Settings) -> None:
    with TestClient(create_app(settings.model_copy(update={"seed_demo": True}))) as c:
        projects = c.get("/api/projects").json()
        assert len(projects) == 1 and "Creator Margin" in projects[0]["name"]
