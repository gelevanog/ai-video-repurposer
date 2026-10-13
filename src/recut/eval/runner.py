"""The evaluation run: every test-set item goes through the real pipeline, then each output is scored.

Per item: caption WER against the reference transcript, speaker accuracy (synthetic), chapter boundaries against
the gold chapters (synthetic), clips (length, cut checks from timestamps and from the audio level, re-transcription
of every rendered clip to catch clipped words, planted-highlight recall, a blind LLM judge of standalone-ness with
random windows and the planted highlights as calibration), framing (face inside the 9:16 window, tracked vs a
centre crop), render time and model calls. Results: results/<run>/items/<id>.json and results/<run>/run.json."""

from __future__ import annotations

import itertools
import json
import random
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from recut.asr.whisper import transcribe_wav
from recut.chapters import youtube_ready
from recut.config import REPO_ROOT, Settings
from recut.eval import testset
from recut.eval.judge import JudgeItem, judge
from recut.llm.factory import model_for
from recut.logging_config import get_logger
from recut.media.ffmpeg import extract_audio, sample_frames
from recut.models import Clip, CropKey, Project
from recut.pipeline import Pipeline
from recut.render import crop
from recut.store import Store
from recut.text import normalize_for_wer, wer

log = get_logger(__name__)


def results_dir(name: str) -> Path:
    return REPO_ROOT / "results" / name


# ------------------------------------------------------------------------------------------------ metrics
def speaker_accuracy(project: Project, gold: dict[str, Any]) -> dict[str, Any]:
    """Word-level speaker accuracy under the best one-to-one mapping of predicted to gold labels."""
    assert project.transcript is not None
    sentences = gold["sentences"]
    pairs: list[tuple[str, str]] = []
    j = 0
    for w in project.transcript.words:
        mid = (w.start + w.end) / 2
        while j < len(sentences) - 1 and sentences[j]["end"] < mid:
            j += 1
        s = sentences[j]
        if s["start"] - 0.3 <= mid <= s["end"] + 0.3:
            pairs.append((w.speaker, s["speaker"]))
    gold_labels = sorted({g for _, g in pairs})
    pred_labels = sorted({p for p, _ in pairs})
    best = 0
    for perm in itertools.permutations(
        gold_labels + ["?"] * max(0, len(pred_labels) - len(gold_labels)), len(pred_labels)
    ):
        mapping = dict(zip(pred_labels, perm, strict=True))
        best = max(best, sum(1 for p, g in pairs if mapping[p] == g))
    return {
        "gold_speakers": len(gold["speakers"]),
        "predicted_speakers": len(project.transcript.speakers),
        "count_correct": len(gold["speakers"]) == len(project.transcript.speakers),
        "word_accuracy": round(best / len(pairs), 4) if pairs else None,
        "words": len(pairs),
        "note": project.transcript.diarization_note,
    }


def chapter_scores(predicted: list[float], gold: list[float], duration: float) -> dict[str, Any]:
    """Boundary precision/recall/F1 at +-15 s and +-30 s (the 00:00 chapter excluded: it is always right)."""
    pred = sorted(t for t in predicted if t > 1.0)
    ref = sorted(t for t in gold if t > 1.0)
    out: dict[str, Any] = {"predicted": len(pred) + 1, "gold": len(ref) + 1}
    for tol in (15, 30):
        used: set[int] = set()
        hits = 0
        errors = []
        for t in ref:
            best = None
            for i, p in enumerate(pred):
                if i in used or abs(p - t) > tol:
                    continue
                if best is None or abs(p - t) < abs(pred[best] - t):
                    best = i
            if best is not None:
                used.add(best)
                hits += 1
                errors.append(abs(pred[best] - t))
        precision = hits / len(pred) if pred else 0.0
        recall = hits / len(ref) if ref else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"tol{tol}"] = {
            "hits": hits,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "mean_abs_error_s": round(sum(errors) / len(errors), 1) if errors else None,
        }
    _ = duration
    return out


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def highlight_recall(clips: list[Clip], highlights: list[dict[str, Any]]) -> dict[str, Any]:
    hit = 0
    per = []
    for h in highlights:
        span = (h["start"], h["end"])
        best = max((overlap(span, (c.start, c.end)) for c in clips), default=0.0)
        frac = best / max(1e-6, span[1] - span[0])
        per.append(round(frac, 2))
        hit += frac >= 0.5
    clip_on_highlight = sum(
        1
        for c in clips
        if max((overlap((c.start, c.end), (h["start"], h["end"])) for h in highlights), default=0.0)
        / max(1e-6, c.end - c.start)
        >= 0.5
    )
    return {
        "highlights": len(highlights),
        "found": hit,
        "recall": round(hit / len(highlights), 3) if highlights else None,
        "coverage": per,
        "clips_mostly_on_a_highlight": clip_on_highlight,
    }


def clip_text(project: Project, clip: Clip) -> str:
    return " ".join(s.text for s in project.sentences[clip.first_sentence : clip.last_sentence + 1])


def random_windows(project: Project, n: int, settings: Settings, seed: int) -> list[tuple[int, int]]:
    """Random sentence windows of 25-55 s: the baseline the judge should reject most of the time."""
    rng = random.Random(seed)
    sents = project.sentences
    out: list[tuple[int, int]] = []
    for _ in range(n * 20):
        if len(out) >= n or len(sents) < 3:
            break
        i = rng.randrange(0, len(sents) - 1)
        target = rng.uniform(25, 55)
        j = i
        while j + 1 < len(sents) and sents[j].end - sents[i].start < target:
            j += 1
        if settings.clip_min_seconds <= sents[j].end - sents[i].start <= settings.clip_max_seconds:
            out.append((i, j))
    return out


def boundary_check(project: Project, clip: Clip, rendered: Path, settings: Settings) -> dict[str, Any]:
    """Re-transcribe the rendered clip: are its first and last spoken words the ones the clip should start and end
    with? A cut inside a word shows up as a missing or garbled boundary word. Independent of the cut logic."""
    assert project.transcript is not None
    expected = [
        t
        for w in project.transcript.words
        if clip.start <= (w.start + w.end) / 2 <= clip.end
        for t in normalize_for_wer(w.text)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_audio(rendered, Path(tmp) / "clip.wav", ffmpeg=settings.ffmpeg)
        hyp = transcribe_wav(wav, settings)
    got = [t for w in hyp.words for t in normalize_for_wer(w.text)]
    if not expected or not got:
        return {"first_ok": False, "last_ok": False, "expected": expected[:2], "got": got[:3]}

    def near(word: str, window: list[str]) -> bool:
        return any(word == g or fuzz.ratio(word, g) >= 80 for g in window)

    return {
        "first_ok": near(expected[0], got[:3]),
        "last_ok": near(expected[-1], got[-3:]),
        "expected": [expected[0], expected[-1]],
        "got": [got[0], got[-1]],
        "clip_wer": round(wer(" ".join(expected), " ".join(got))["wer"], 3),
    }


def framing(project: Project, clip: Clip, settings: Settings) -> dict[str, Any] | None:
    if clip.crop is None or clip.crop.layout != "track" or not project.source.has_video:
        return None
    detector = crop.FaceDetector(settings.face_model)
    frames = sample_frames(
        Path(project.source.path),
        start=clip.start,
        end=clip.end,
        fps=2.0,
        width=min(640, project.source.width),
        src_width=project.source.width,
        src_height=project.source.height,
    )
    samples = crop.analyse(frames, detector)
    window = min(1.0, project.source.height * 9 / 16 / max(1, project.source.width))
    return {
        "samples_with_face": sum(1 for s in samples if s.faces),
        "tracked": round(crop.coverage(samples, clip.crop.keys, window_frac=window), 3),
        "center": round(crop.coverage(samples, [CropKey(t=0, cx=0.5)], window_frac=window), 3),
        "keys": len(clip.crop.keys),
    }


def item_wer(item: dict[str, Any], project: Project) -> dict[str, Any]:
    assert project.transcript is not None
    words = [(w.start, w.end, w.text) for w in project.transcript.words]
    ref, hyp, offset = testset.aligned_texts(item, words, project.source.duration)
    result: dict[str, Any] = dict(wer(ref, hyp))
    result["caption_offset_s"] = offset
    return result


def rescore(name: str) -> None:
    """Recompute the WER of every item of a run from its saved project (after a change in the scoring)."""
    store = Store(REPO_ROOT / ".cache" / "eval-projects" / name)
    manifest = {i["id"]: i for i in testset.load_manifest()}
    for path in sorted((results_dir(name) / "items").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        project = store.load(row["project_id"])
        row["wer"] = item_wer(manifest[row["id"]], project)
        path.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{row['id']}: WER {row['wer']['wer']:.3f} (captions shifted {row['wer']['caption_offset_s']:+.0f} s)")


def rejudge(name: str, settings: Settings, only: list[str] | None = None) -> None:
    """Ask the judge again for items whose verdicts failed (rate limits, a cut-off reply)."""
    store = Store(REPO_ROOT / ".cache" / "eval-projects" / name)
    manifest = {i["id"]: i for i in testset.load_manifest()}
    for path in sorted((results_dir(name) / "items").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        verdicts = row.get("judge") or {}
        if only and row["id"] not in only:
            continue
        cloud = [v for v in verdicts.values() if "standalone" in v and not str(v.get("judge", "")).startswith("fake")]
        if not only and verdicts and len(cloud) == len(verdicts):
            continue
        project = store.load(row["project_id"])
        items = [JudgeItem(c["id"], "model", c["text"]) for c in row["clips"]]
        seed = zlib.crc32(row["id"].encode()) % 1000
        for k, (i, j) in enumerate(random_windows(project, len(row["clips"]), settings, seed=seed)):
            items.append(JudgeItem(f"random-{k}", "random", " ".join(s.text for s in project.sentences[i : j + 1])))
        if manifest[row["id"]]["kind"] == "synthetic":
            gold = json.loads(testset.gold_path(manifest[row["id"]]).read_text(encoding="utf-8"))
            for k, h in enumerate(gold["highlights"]):
                items.append(JudgeItem(f"gold-{k}", "gold", h["text"]))
        row["judge"] = judge(items, settings, tag=f"eval-{name}:judge", seed=8)
        path.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        ok = sum(1 for v in row["judge"].values() if "standalone" in v)
        print(f"{row['id']}: {ok} of {len(items)} verdicts")


def posts_summary(project: Project) -> dict[str, Any]:
    o = project.outputs
    return {
        "social_posts": len(o.social),
        "within_limit": sum(1 for p in o.social if p.within_limit),
        "truncated": sum(1 for p in o.social if p.truncated),
        "flags_social": sum(1 for p in o.social for i in p.issues if i.kind in ("number", "quote")),
        "flags_long_form": sum(1 for i in o.issues if i.kind in ("number", "quote")),
        "flag_texts": [i.text for p in o.social for i in p.issues if i.kind in ("number", "quote")]
        + [i.text for i in o.issues if i.kind in ("number", "quote")],
        "title_chars": len(o.seo_title),
        "description_chars": len(o.seo_description),
        "blog_words": len(o.blog_post.split()),
        "show_notes_words": len(o.show_notes.split()),
        "chapters_youtube_ready": not youtube_ready(project.chapters, project.source.duration),
        "stage_message": project.stage("posts").message,
    }


def refresh_chapters(name: str, settings: Settings) -> None:
    """Re-run the chapters stage of every project of a run and rescore the chapters against the gold data."""
    work = REPO_ROOT / ".cache" / "eval-projects" / name
    pipeline = Pipeline(
        settings.model_copy(update={"data_dir": work}),
        models=lambda tag: model_for(settings, tag=f"eval-{name}:{tag}"),
    )
    manifest = {i["id"]: i for i in testset.load_manifest()}
    for path in sorted((results_dir(name) / "items").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        project = pipeline.process(row["project_id"], stages=("chapters",), force=True)
        row["chapters_out"] = [{"start": c.start, "title": c.title} for c in project.chapters]
        item = manifest[row["id"]]
        if item["kind"] == "synthetic":
            gold = json.loads(testset.gold_path(item).read_text(encoding="utf-8"))
            row["chapters"] = chapter_scores(
                [c.start for c in project.chapters], [c["start"] for c in gold["chapters"]], project.source.duration
            )
        if "posts" in row:
            row["posts"]["chapters_youtube_ready"] = not youtube_ready(project.chapters, project.source.duration)
        path.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{row['id']}: {len(project.chapters)} chapters")


def recheck_posts(name: str) -> None:
    """Re-run only the deterministic checks (limits, grounding) on the texts a run already generated, without any
    model call, and write the posts summary into each item."""
    from recut.content import limits
    from recut.content.grounding import TranscriptIndex, check

    store = Store(REPO_ROOT / ".cache" / "eval-projects" / name)
    for path in sorted((results_dir(name) / "items").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        project = store.load(row["project_id"])
        index = TranscriptIndex(" ".join(s.text for s in project.sentences), extra=project.source.title)
        o = project.outputs
        for post in o.social:
            spec = limits.PLATFORMS[post.platform]
            post.chars = limits.count(post.text, spec.counting)
            post.within_limit = post.chars <= spec.limit
            post.issues = [i for i in post.issues if i.kind == "length"] + check(post.text, index)
        credits = [x for x in (project.source.attribution, project.source.license) if x]

        def strip(text: str, credits: list[str] = credits) -> str:
            for credit in credits:
                text = text.replace(credit, " ")
            return text

        description = strip(o.seo_description.split("\n\nChapters:")[0])
        notes = strip(o.show_notes.split("## Quotes")[0])
        blog = strip(o.blog_post)
        o.issues = [i for i in o.issues if i.kind == "length"]
        for label, text in (
            ("title", o.seo_title),
            ("description", description),
            ("show notes", notes),
            ("blog post", blog),
        ):
            for issue in check(text, index):
                o.issues.append(issue.model_copy(update={"message": f"{label}: {issue.message}"}))
        store.save(project)
        row["posts"] = posts_summary(project)
        path.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{row['id']}: {row['posts']['social_posts']} posts, flags {row['posts']['flag_texts']}")


def refresh_posts(name: str, settings: Settings) -> None:
    """Re-run the posts stage of every project of a run (after a fix in the posts code) and update the items."""
    work = REPO_ROOT / ".cache" / "eval-projects" / name
    pipeline = Pipeline(
        settings.model_copy(update={"data_dir": work}),
        models=lambda tag: model_for(settings, tag=f"eval-{name}:{tag}"),
    )
    for path in sorted((results_dir(name) / "items").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        project = pipeline.store.load(row["project_id"]) if pipeline.store.exists(row["project_id"]) else None
        if project is None:
            continue
        project = pipeline.process(project.id, stages=("posts",), force=True)
        row["posts"] = posts_summary(project)
        path.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{row['id']}: {row['posts']['social_posts']} posts, {row['posts']['flags_social']} flags")


# ------------------------------------------------------------------------------------------------ run
def evaluate_item(
    item: dict[str, Any],
    name: str,
    settings: Settings,
    *,
    render: bool,
    run_judge: bool,
    judge_cloud: bool | None = None,
) -> dict[str, Any]:
    media = testset.media_path(item)
    work = REPO_ROOT / ".cache" / "eval-projects" / name
    pipeline = Pipeline(
        settings.model_copy(update={"data_dir": work}),
        models=lambda tag: model_for(settings, tag=f"eval-{name}:{tag}"),
    )
    started = time.monotonic()
    project = pipeline.create(
        media, name=item["title"], license=item["license"], attribution=item.get("attribution", ""), copy=False
    )
    project = pipeline.process(project.id)
    assert project.transcript is not None
    process_seconds = time.monotonic() - started
    duration = project.source.duration
    row: dict[str, Any] = {
        "id": item["id"],
        "kind": item["kind"],
        "title": item["title"],
        "duration": duration,
        "project_id": project.id,
        "model": project.llm,
        "stages": {s.name: {"seconds": s.seconds, "message": s.message} for s in project.stages},
        "asr_seconds": project.transcript.seconds,
        "asr_rtf": round(project.transcript.seconds / max(1.0, duration), 4),
        "process_seconds_this_run": round(process_seconds, 1),
    }
    row["wer"] = item_wer(item, project)
    gold: dict[str, Any] | None = None
    if item["kind"] == "synthetic":
        gold = json.loads(testset.gold_path(item).read_text(encoding="utf-8"))
        row["speakers"] = speaker_accuracy(project, gold)
        row["chapters"] = chapter_scores(
            [c.start for c in project.chapters], [c["start"] for c in gold["chapters"]], duration
        )
        row["highlights"] = highlight_recall(project.clips, gold["highlights"])
    row["chapters_out"] = [{"start": c.start, "title": c.title} for c in project.chapters]
    row["posts"] = posts_summary(project)
    clips_out = []
    for clip in project.clips:
        entry: dict[str, Any] = {
            "id": clip.id,
            "rank": clip.rank,
            "start": clip.start,
            "end": clip.end,
            "duration": clip.duration,
            "title": clip.title,
            "score": clip.score,
            "scores": clip.scores.model_dump(),
            "issues": [i.code for i in clip.issues],
            "cut_in": clip.cut_in.model_dump() if clip.cut_in else None,
            "cut_out": clip.cut_out.model_dump() if clip.cut_out else None,
            "adjustments": clip.adjustments,
            "text": clip_text(project, clip),
        }
        if render:
            project = pipeline.render(project.id, clip.id)
            fresh = project.clip(clip.id)
            assert fresh.render is not None
            entry["render"] = {
                "seconds": fresh.render.seconds,
                "layout": fresh.render.layout,
                "seconds_per_output_minute": round(fresh.render.seconds / max(1e-6, fresh.render.duration) * 60, 2),
            }
            entry["boundary"] = boundary_check(project, fresh, Path(fresh.render.path), settings)
            entry["framing"] = framing(project, fresh, settings)
        clips_out.append(entry)
    row["clips"] = clips_out
    if run_judge and project.clips:
        items = [JudgeItem(c["id"], "model", c["text"]) for c in clips_out]
        for k, (i, j) in enumerate(
            random_windows(project, len(clips_out), settings, seed=zlib.crc32(item["id"].encode()) % 1000)
        ):
            items.append(JudgeItem(f"random-{k}", "random", " ".join(s.text for s in project.sentences[i : j + 1])))
        if gold is not None:
            for k, h in enumerate(gold["highlights"]):
                items.append(JudgeItem(f"gold-{k}", "gold", h["text"]))
        row["judge"] = judge(items, settings, tag=f"eval-{name}:judge", cloud=judge_cloud)
    return row


def run(
    *,
    name: str,
    settings: Settings,
    only: list[str] | None = None,
    render: bool = True,
    judge: bool = True,
    judge_cloud: bool | None = None,
) -> None:
    out = results_dir(name) / "items"
    out.mkdir(parents=True, exist_ok=True)
    items = [i for i in testset.load_manifest() if not only or i["id"] in only]
    missing = [i["id"] for i in items if not testset.media_path(i).exists()]
    if missing:
        raise SystemExit(f"missing media (run `recut testset download` / `recut testset synth`): {', '.join(missing)}")
    for item in items:
        target = out / f"{item['id']}.json"
        if target.exists() and not only:
            print(f"{item['id']}: done already")
            continue
        started = time.monotonic()
        row = evaluate_item(item, name, settings, render=render, run_judge=judge, judge_cloud=judge_cloud)
        row["eval_seconds"] = round(time.monotonic() - started, 1)
        target.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{item['id']}: {len(row['clips'])} clips, WER {row['wer']['wer']:.3f}, {row['eval_seconds']:.0f} s")
    meta = {
        "run": name,
        "provider": settings.llm_provider,
        "model": settings.model_name,
        "fallbacks": settings.fallbacks,
        "judge": settings.judge_model
        if (judge_cloud if judge_cloud is not None else settings.llm_provider != "fake")
        else "fake/rules",
        "asr": f"faster-whisper {settings.whisper_model} ({settings.whisper_compute_type}, CPU)",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (results_dir(name) / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
