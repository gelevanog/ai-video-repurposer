"""Command line: process a video end to end, render clips, export, run the API, the MCP server and the evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from recut import __version__
from recut.config import LLMProviderKind, Settings, get_settings
from recut.logging_config import configure_logging
from recut.text import ts

app = typer.Typer(help="Recut: long video -> short clips, captions, chapters and posts.", no_args_is_help=True)
testset_app = typer.Typer(help="Build or fetch the evaluation test set.", no_args_is_help=True)
eval_app = typer.Typer(help="Run the evaluation (see README > Evaluation).", no_args_is_help=True)
app.add_typer(testset_app, name="testset")
app.add_typer(eval_app, name="eval")
console = Console()


def _settings(provider: LLMProviderKind | None = None, model: str | None = None) -> Settings:
    settings = get_settings()
    updates: dict[str, object] = {}
    if provider:
        updates["llm_provider"] = provider
    if model:
        updates["llm_model"] = model
    return settings.model_copy(update=updates) if updates else settings


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING")] = "WARNING",
    version: Annotated[bool, typer.Option("--version", help="Print the version.")] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()
    configure_logging(log_level, get_settings().log_format)


@app.command()
def process(
    source: Annotated[Path, typer.Argument(help="A local video or audio file.", exists=True, dir_okay=False)],
    name: Annotated[str, typer.Option(help="Project name (default: the file name).")] = "",
    license: Annotated[str, typer.Option(help='Licence of the source, e.g. "CC BY 4.0", "Public domain".')] = "",
    attribution: Annotated[str, typer.Option(help="Credit line added to descriptions and posts.")] = "",
    source_url: Annotated[str, typer.Option(help="Where the source came from.")] = "",
    copy: Annotated[bool, typer.Option(help="Copy the file into the project folder.")] = True,
    provider: Annotated[LLMProviderKind | None, typer.Option(help="fake, openrouter or compat.")] = None,
    model: Annotated[str | None, typer.Option(help="Model id (openrouter: must end in :free).")] = None,
    render_top: Annotated[int, typer.Option(help="Render the N best clips after processing.")] = 0,
) -> None:
    """Transcribe, find clips, write chapters and posts. Nothing is published: review in the UI or with `recut show`."""
    from recut.pipeline import Pipeline

    pipeline = Pipeline(_settings(provider, model))
    project = pipeline.create(
        source, name=name, license=license, attribution=attribution, source_url=source_url, copy=copy
    )
    console.print(f"project [bold]{project.id}[/bold]")

    def progress(p: object) -> None:
        from recut.models import Project

        assert isinstance(p, Project)
        running = [s for s in p.stages if s.state == "running"]
        if running:
            console.print(f"  … {running[0].name}")

    project = pipeline.process(project.id, on_stage=progress)
    for clip in project.clips[:render_top]:
        console.print(f"  rendering {clip.id} ({clip.duration:.0f} s)")
        project = pipeline.render(project.id, clip.id)
    _print_project(project.id)


def _print_project(project_id: str) -> None:
    from recut.pipeline import Pipeline

    project = Pipeline(get_settings()).store.load(project_id)
    table = Table(title=f"{project.name} · {ts(project.source.duration)} · {project.llm or 'no model yet'}")
    for column in ("#", "score", "in", "out", "len", "title", "status", "issues"):
        table.add_column(column)
    for clip in project.clips:
        table.add_row(
            str(clip.rank),
            f"{clip.score:.0f}",
            ts(clip.start, millis=True),
            ts(clip.end, millis=True),
            f"{clip.duration:.1f}s",
            clip.title,
            clip.status + (" · rendered" if clip.render else ""),
            ", ".join(i.code for i in clip.issues if i.severity != "info") or "-",
        )
    console.print(table)
    if project.chapters_text:
        console.print("[bold]Chapters[/bold]\n" + project.chapters_text)


@app.command()
def show(project_id: str) -> None:
    """Clips, chapters and stage timings of a project."""
    _print_project(project_id)


@app.command("list")
def list_projects() -> None:
    """All projects."""
    from recut.store import Store

    for project in Store(get_settings().data_dir).list():
        console.print(
            f"{project.id}  {project.status:<10} {ts(project.source.duration):>8}  {len(project.clips)} clips  {project.name}"
        )


@app.command()
def transcribe(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[
        Path | None, typer.Option("-o", help="Write the transcript JSON here (default: <file>.words.json).")
    ] = None,
    diarize: Annotated[bool, typer.Option(help="Separate speakers.")] = True,
) -> None:
    """Speech to text with word timestamps (faster-whisper on CPU), optionally with speakers."""
    import tempfile

    from recut.asr import diarize as diarization
    from recut.asr.whisper import sidecar_path, transcribe_wav
    from recut.media.ffmpeg import extract_audio, load_wav, probe

    settings = get_settings()
    info = probe(source)
    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_audio(source, Path(tmp) / "audio.wav")
        transcript = transcribe_wav(wav, settings, duration=info.duration)
        if diarize:
            result = diarization.diarize(transcript, load_wav(wav), max_speakers=settings.max_speakers)
            transcript = diarization.apply(transcript, result)
    target = out or sidecar_path(source)
    target.write_text(transcript.model_dump_json(), encoding="utf-8")
    console.print(
        f"{len(transcript.words)} words, speakers {', '.join(transcript.speakers)}, "
        f"{transcript.seconds:.1f} s for {info.duration:.0f} s of audio -> {target}"
    )


@app.command()
def render(
    project_id: str,
    clip_id: Annotated[str | None, typer.Argument(help="Clip id (default: the best clip).")] = None,
    top: Annotated[int, typer.Option(help="Render the N best clips instead.")] = 0,
) -> None:
    """Render clips to 1080x1920 MP4 with burned-in captions."""
    from recut.pipeline import Pipeline

    pipeline = Pipeline(get_settings())
    project = pipeline.store.load(project_id)
    ids = [c.id for c in project.clips[:top]] if top else [clip_id or project.clips[0].id]
    for cid in ids:
        project = pipeline.render(project_id, cid)
        info = project.clip(cid).render
        assert info is not None
        console.print(f"{cid}: {info.path} ({info.duration:.1f} s video in {info.seconds:.1f} s, {info.layout})")


@app.command()
def rerun(
    project_id: str,
    stage: Annotated[str, typer.Argument(help="clips, chapters or posts")],
    provider: Annotated[LLMProviderKind | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Run one model stage again (e.g. with another model)."""
    from recut.pipeline import Pipeline

    if stage not in ("clips", "chapters", "posts"):
        raise typer.BadParameter("stage must be clips, chapters or posts")
    pipeline = Pipeline(_settings(provider, model))
    pipeline.process(project_id, stages=(stage,), force=True)  # type: ignore[arg-type]
    _print_project(project_id)


@app.command()
def export(project_id: str, out: Annotated[Path, typer.Option("-o")] = Path("recut-export.zip")) -> None:
    """Zip the approved clips, captions (SRT), chapters and approved texts."""
    from recut.pipeline import Pipeline

    data = Pipeline(get_settings()).export_bundle(project_id)
    out.write_bytes(data)
    console.print(f"{out} ({len(data) / 1e6:.1f} MB)")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """The API (and the job runner) for the web UI."""
    import uvicorn

    uvicorn.run("recut.api.app:create_default_app", factory=True, host=host, port=port, log_level="warning")


@app.command()
def mcp(
    http: Annotated[bool, typer.Option(help="Streamable HTTP instead of stdio.")] = False,
    port: Annotated[int, typer.Option()] = 8765,
) -> None:
    """MCP server for Claude Code / Claude Desktop."""
    from recut.mcp_server import serve as serve_mcp

    serve_mcp(http=http, port=port)


# ---------------------------------------------------------------------------------------------- test set
@testset_app.command("scripts")
def testset_scripts(
    only: Annotated[str, typer.Option(help="Comma-separated ids.")] = "",
    model: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Draft the synthetic talk scripts from data/testset/synthetic/outlines.json with a free model."""
    from recut.config import DEFAULT_SCRIPT_MODEL
    from recut.eval import scripts

    settings = _settings("openrouter", model or DEFAULT_SCRIPT_MODEL).model_copy(
        update={"llm_fallback_models": ["nvidia/nemotron-3-ultra-550b-a55b:free"]}
    )
    for path in scripts.draft_all(settings, only=[x for x in only.split(",") if x]):
        console.print(f"{path}")


@testset_app.command("synth")
def testset_synth(only: Annotated[str, typer.Option(help="Comma-separated ids.")] = "") -> None:
    """Render the synthetic talks (Kokoro voices + slides) into data/testset/media/ with exact gold data."""
    from recut.eval import testset

    for row in testset.synthesize(only=[x for x in only.split(",") if x]):
        console.print(json.dumps(row))


@testset_app.command("download")
def testset_download() -> None:
    """Download the public-domain NASA videos and captions listed in data/testset/manifest.json (checksums verified)."""
    from recut.eval import testset

    for row in testset.download():
        console.print(json.dumps(row))


@testset_app.command("verify")
def testset_verify() -> None:
    """Check every test-set file against the manifest's SHA-256."""
    from recut.eval import testset

    bad = testset.verify()
    for row in bad:
        console.print(json.dumps(row))
    raise typer.Exit(1 if bad else 0)


# ---------------------------------------------------------------------------------------------- evaluation
@eval_app.command("smoke")
def eval_smoke(models: Annotated[str, typer.Option(help="Comma-separated :free model ids.")]) -> None:
    """Smoke-test candidate free models on the bundled sample (results/smoke.json)."""
    from recut.eval.smoke import run

    run(
        [m for m in models.split(",") if m],
        get_settings().samples_dir / "creator-margin-ep12.mp4.words.json",
        get_settings(),
        Path("results/smoke.json"),
    )


@eval_app.command("run")
def eval_run(
    name: Annotated[str, typer.Option(help="Run name (results/<name>/).")] = "main",
    only: Annotated[str, typer.Option(help="Comma-separated item ids.")] = "",
    provider: Annotated[LLMProviderKind | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option()] = None,
    render: Annotated[bool, typer.Option(help="Render every suggested clip and check it.")] = True,
    judge: Annotated[bool, typer.Option(help="Ask the judge model whether each clip stands alone.")] = True,
) -> None:
    """Process every test-set item and score transcripts, speakers, chapters, clips, cuts and render time."""
    from recut.eval import runner

    runner.run(
        name=name,
        settings=_settings(provider, model),
        only=[x for x in only.split(",") if x],
        render=render,
        judge=judge,
    )


@eval_app.command("summary")
def eval_summary(names: Annotated[str, typer.Option(help="Comma-separated run names.")] = "main,rules") -> None:
    """Fold results/<run>/items/*.json into results/summary.json and print the README table."""
    from recut.eval import report

    report.summarize([n for n in names.split(",") if n])


if __name__ == "__main__":  # pragma: no cover
    app()
