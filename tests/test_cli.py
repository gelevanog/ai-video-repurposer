from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from recut import cli
from recut.config import Settings
from tests.conftest import needs_ffmpeg

pytestmark = needs_ffmpeg
runner = CliRunner()


@pytest.fixture(autouse=True)
def offline(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    import recut.config

    monkeypatch.setattr(recut.config, "get_settings", lambda: settings)


def test_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0 and "0.1.0" in result.output


def test_process_list_show_and_export_guard(sample: Path) -> None:
    result = runner.invoke(cli.app, ["process", str(sample), "--license", "CC0", "--name", "Demo talk"])
    assert result.exit_code == 0, result.output
    assert "Chapters" in result.output and "score" in result.output
    listed = runner.invoke(cli.app, ["list"])
    assert "Demo talk" in listed.output and "ready" in listed.output
    pid = listed.output.split()[0]
    assert runner.invoke(cli.app, ["show", pid]).exit_code == 0
    export = runner.invoke(cli.app, ["export", pid, "-o", str(sample.parent / "x.zip")])
    assert export.exit_code != 0 and "nothing is approved" in str(export.exception)


def test_rerun_validates_the_stage(sample: Path) -> None:
    result = runner.invoke(cli.app, ["rerun", "whatever-000000", "transcribe"])
    assert result.exit_code != 0
