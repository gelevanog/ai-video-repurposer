"""CPU timing benchmark on an otherwise idle machine: speech recognition (no transcript cache), speaker
separation, face-tracking analysis and rendering, per minute of media. Writes results/bench.json.

The evaluation run measures the same steps, but it shares the CPU with whatever else runs (text-to-speech for the
test set, other renders); these numbers are the ones to plan with."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from recut.asr import diarize as diarization
from recut.asr.whisper import transcribe_wav
from recut.config import REPO_ROOT, Settings
from recut.eval import testset
from recut.media.ffmpeg import extract_audio, load_wav, probe
from recut.models import Project
from recut.pipeline import Pipeline
from recut.store import Store


def _load() -> list[float]:
    return [round(x, 2) for x in os.getloadavg()]


def run(settings: Settings, *, items: list[str], run: str = "main", clips_per_item: int = 3) -> dict[str, Any]:
    manifest = {i["id"]: i for i in testset.load_manifest()}
    store = Store(REPO_ROOT / ".cache" / "eval-projects" / run)
    projects = {Path(p.source.path).stem: p for p in store.list()}
    rows: list[dict[str, Any]] = []
    out: dict[str, Any] = {
        "cpu": os.cpu_count(),
        "load_before": _load(),
        "whisper": f"{settings.whisper_model} {settings.whisper_compute_type}",
        "render": f"{settings.render_width}x{settings.render_height} libx264 {settings.render_preset} crf {settings.render_crf}",
        "items": rows,
    }
    pipeline = Pipeline(settings.model_copy(update={"data_dir": store.root}))
    for item_id in items:
        item = manifest[item_id]
        media = testset.media_path(item)
        info = probe(media)
        work = REPO_ROOT / ".cache" / "bench"
        work.mkdir(parents=True, exist_ok=True)
        t = time.monotonic()
        wav = extract_audio(media, work / f"{item_id}.wav")
        audio_s = time.monotonic() - t
        t = time.monotonic()
        transcript = transcribe_wav(wav, settings, duration=info.duration)
        asr_s = time.monotonic() - t
        t = time.monotonic()
        result = diarization.diarize(transcript, load_wav(wav), max_speakers=settings.max_speakers)
        diar_s = time.monotonic() - t
        row: dict[str, Any] = {
            "id": item_id,
            "duration_s": info.duration,
            "resolution": f"{info.width}x{info.height}",
            "audio_extract_s": round(audio_s, 1),
            "asr_s": round(asr_s, 1),
            "asr_s_per_min": round(asr_s / info.duration * 60, 2),
            "diarize_s": round(diar_s, 1),
            "speakers": len(result.speakers),
            "renders": [],
        }
        project: Project | None = projects.get(item_id)
        if project is not None:
            for clip in project.clips[:clips_per_item]:
                # a fresh render: the framing is planned again (face detection included in the time)
                fresh = store.load(project.id)
                fresh.clip(clip.id).crop = None
                store.save(fresh)
                started = time.monotonic()
                done = pipeline.render(project.id, clip.id)
                seconds = time.monotonic() - started
                rendered = done.clip(clip.id)
                row["renders"].append(
                    {
                        "clip": clip.id,
                        "duration_s": rendered.duration,
                        "layout": rendered.render.layout if rendered.render else "",
                        "seconds": round(seconds, 1),
                        "s_per_output_min": round(seconds / rendered.duration * 60, 1),
                    }
                )
        rows.append(row)
        print(json.dumps(row))
    total_media = sum(r["duration_s"] for r in rows)
    renders = [x for r in rows for x in r["renders"]]
    out["load_after"] = _load()
    out["asr_s_per_source_min"] = round(sum(r["asr_s"] for r in rows) / total_media * 60, 2)
    out["diarize_s_per_source_min"] = round(sum(r["diarize_s"] for r in rows) / total_media * 60, 2)
    out["render_s_per_output_min"] = (
        round(sum(x["seconds"] for x in renders) / sum(x["duration_s"] for x in renders) * 60, 1) if renders else None
    )
    (REPO_ROOT / "results" / "bench.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
