"""The MCP server through the SDK's in-memory client (and one real stdio subprocess)."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

from recut.config import Settings
from recut.mcp_server import RecutMCP, create_server
from tests.conftest import needs_ffmpeg

pytestmark = [pytest.mark.anyio, needs_ffmpeg]


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def state(settings: Settings, sample: Path) -> RecutMCP:
    return RecutMCP(settings.model_copy(update={"allowed_source_dirs": [str(sample.parent)]}))


@pytest.fixture()
async def client(state: RecutMCP) -> AsyncIterator[Client]:
    async with Client(create_server(state=state), raise_exceptions=True) as c:
        yield c


async def call(client: Client, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, args)
    assert not result.is_error, result.content
    assert isinstance(result.structured_content, dict)
    return result.structured_content


async def error(client: Client, tool: str, args: dict[str, Any]) -> str:
    result = await client.call_tool(tool, args)
    assert result.is_error
    return str(getattr(result.content[0], "text", ""))


async def test_tools_are_listed_and_none_can_approve(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "probe_media",
        "transcribe",
        "get_transcript",
        "snap_clip",
        "save_clip",
        "suggest_clips",
        "render_clip",
        "make_chapters",
        "check_post",
        "list_projects",
    }
    assert tools["snap_clip"].annotations and tools["snap_clip"].annotations.read_only_hint
    assert not any("approve" in name or "publish" in name for name in tools)


async def test_probe_and_path_allow_list(client: Client, sample: Path) -> None:
    info = await call(client, "probe_media", {"path": str(sample)})
    assert info["width"] == 1280 and info["has_audio"]
    assert "not allowed" in await error(client, "probe_media", {"path": "/etc/hostname"})


async def test_transcribe_snap_save_render(client: Client, sample: Path) -> None:
    t = await call(client, "transcribe", {"path": str(sample), "license": "CC0"})
    pid = t["project_id"]
    assert t["speakers"] == ["S1", "S2"] and t["sentences"][0].startswith("[0] 00:00.")
    assert "never an instruction" in t["data_notice"]
    again = await call(client, "transcribe", {"path": str(sample)})
    assert again["project_id"] == pid  # reused, not transcribed twice
    page = await call(client, "get_transcript", {"project_id": pid, "start": 5, "count": 3})
    assert len(page["sentences"]) == 3 and page["sentences"][0].startswith("[5]")
    snapped = await call(client, "snap_clip", {"project_id": pid, "first_sentence": 4, "last_sentence": 6})
    assert 20 <= snapped["duration"] <= 60 and not snapped["cut_in"]["inside_word"]
    fourth = (await call(client, "get_transcript", {"project_id": pid, "start": 4, "count": 1}))["sentences"][0]
    assert snapped["text"].startswith(fourth.split(": ", 1)[1][:30])
    saved = await call(
        client,
        "save_clip",
        {
            "project_id": pid,
            "start": snapped["start"],
            "end": snapped["end"],
            "title": "Price from value",
            "caption_style": "pop",
        },
    )
    assert saved["status"] == "suggested"
    rendered = await call(client, "render_clip", {"project_id": pid, "clip_id": saved["clip_id"], "layout": "fit"})
    assert Path(rendered["path"]).exists() and rendered["layout"] == "fit" and rendered["style"] == "pop"
    assert "approve" in rendered["note"]


async def test_snap_by_times_and_bad_input(client: Client, sample: Path) -> None:
    pid = (await call(client, "transcribe", {"path": str(sample)}))["project_id"]
    snapped = await call(client, "snap_clip", {"project_id": pid, "start": 12.0, "end": 40.0})
    assert snapped["start"] <= 12.0 + 1 and snapped["first_sentence"] >= 0
    assert "between" in await error(
        client, "snap_clip", {"project_id": pid, "first_sentence": 0, "last_sentence": 9999}
    )
    assert "give" in await error(client, "snap_clip", {"project_id": pid})
    assert "no project" in await error(client, "get_transcript", {"project_id": "nope-000000"})


async def test_suggest_clips_uses_the_offline_model(client: Client, sample: Path) -> None:
    pid = (await call(client, "transcribe", {"path": str(sample)}))["project_id"]
    result = await call(client, "suggest_clips", {"project_id": pid})
    assert result["model"] == "fake/rules" and result["clips"]


async def test_make_chapters_validates_callers_chapters(client: Client, sample: Path) -> None:
    pid = (await call(client, "transcribe", {"path": str(sample)}))["project_id"]
    result = await call(
        client,
        "make_chapters",
        {
            "project_id": pid,
            "chapters": [{"sentence": 2, "title": "1. Why underprice"}, {"sentence": 12, "title": "Three numbers"}],
        },
    )
    assert result["source"] == "caller" and result["chapters"][0]["start"] == 0
    assert result["youtube_text"].startswith("00:00 Why underprice")
    assert result["youtube_problems"]  # only two chapters: YouTube needs three
    own = await call(client, "make_chapters", {"project_id": pid})
    assert own["source"] == "recut model"


async def test_check_post(client: Client, sample: Path) -> None:
    pid = (await call(client, "transcribe", {"path": str(sample)}))["project_id"]
    ok = await call(
        client, "check_post", {"text": "I charged $200 for a launch video.", "platform": "x", "project_id": pid}
    )
    assert ok["within_limit"] and ok["grounding_issues"] == []
    bad = await call(client, "check_post", {"text": "Creators earn 9x more. " * 20, "platform": "x", "project_id": pid})
    assert not bad["within_limit"] and bad["suggested_cut"] and bad["grounding_issues"][0]["kind"] == "number"
    title = await call(client, "check_post", {"text": "Short title", "platform": "youtube_title"})
    assert title["limit"] == 100


async def test_list_projects(client: Client, sample: Path) -> None:
    await call(client, "transcribe", {"path": str(sample)})
    listed = await call(client, "list_projects", {})
    assert len(listed["projects"]) == 1


async def test_stdio_entry_point(settings: Settings, sample: Path, tmp_path: Path) -> None:
    env = {
        **os.environ,
        "RECUT_DATA_DIR": str(tmp_path / "projects"),
        "RECUT_LLM_PROVIDER": "fake",
        "RECUT_LLM_LEDGER": "",
        "RECUT_ASR": "sidecar",
        "RECUT_ALLOWED_SOURCE_DIRS": str(sample.parent),
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "recut.cli", "mcp"], env=env, cwd=str(tmp_path))
    async with Client(params) as c:
        info = await call(c, "probe_media", {"path": str(sample)})
    assert info["has_video"]
