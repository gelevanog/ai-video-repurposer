"""Folds results/<run>/items/*.json and the call ledger into results/summary.json (read by the UI's evaluation
page and quoted in the README)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from recut.config import REPO_ROOT

RESULTS = REPO_ROOT / "results"


def _items(name: str) -> list[dict[str, Any]]:
    folder = RESULTS / name / "items"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))] if folder.exists() else []


def _pct(num: float, den: float) -> float | None:
    return round(100.0 * num / den, 1) if den else None


def _pooled_wer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = sum(r["wer"]["substitutions"] + r["wer"]["deletions"] + r["wer"]["insertions"] for r in rows)
    words = sum(r["wer"]["ref_words"] for r in rows)
    missed = sum(r["wer"]["substitutions"] + r["wer"]["deletions"] for r in rows)
    return {
        "wer": round(errors / words, 4) if words else None,
        "ref_words_wrong_or_missed": round(missed / words, 4) if words else None,
        "ref_words": int(words),
        "items": len(rows),
    }


def ledger_stats(prefix: str) -> dict[str, Any]:
    path = RESULTS / "calls.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if str(row.get("tag", "")).startswith(prefix):
                    rows.append(row)
    ok = [r for r in rows if r.get("status") == "ok"]
    served = Counter(r.get("served_model") for r in ok)
    return {
        "requests": len(rows),
        "ok": len(ok),
        "retries_or_errors": len(rows) - len(ok),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in ok),
        "output_tokens": sum(int(r.get("output_tokens") or 0) for r in ok),
        "median_latency_s": round(median(float(r["latency_s"]) for r in ok), 1) if ok else None,
        "served_models": dict(served),
        "all_free": all(str(m).endswith(":free") for m in served) if served else True,
        "by_tag": dict(Counter(str(r.get("tag", "")).split(":", 1)[-1] for r in rows)),
    }


def summarize_run(name: str) -> dict[str, Any]:
    rows = _items(name)
    if not rows:
        return {}
    meta_path = RESULTS / name / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    synth = [r for r in rows if r["kind"] == "synthetic"]
    real = [r for r in rows if r["kind"] == "nasa"]
    clips = [c for r in rows for c in r["clips"]]
    duration = sum(r["duration"] for r in rows)

    # chapters (synthetic, gold)
    def chap(tol: str) -> dict[str, Any]:
        hits = sum(r["chapters"][tol]["hits"] for r in synth)
        pred = sum(r["chapters"]["predicted"] - 1 for r in synth)
        gold = sum(r["chapters"]["gold"] - 1 for r in synth)
        p = hits / pred if pred else 0.0
        rc = hits / gold if gold else 0.0
        errs = [
            r["chapters"][tol]["mean_abs_error_s"] for r in synth if r["chapters"][tol]["mean_abs_error_s"] is not None
        ]
        return {
            "precision": round(p, 3),
            "recall": round(rc, 3),
            "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0,
            "boundaries_gold": gold,
            "boundaries_predicted": pred,
            "mean_abs_error_s": round(sum(errs) / len(errs), 1) if errs else None,
        }

    # speakers
    multi = [r for r in synth if r["speakers"]["gold_speakers"] > 1]
    single = [r for r in synth if r["speakers"]["gold_speakers"] == 1]
    spk_words = sum(r["speakers"]["words"] for r in multi)
    spk_ok = sum((r["speakers"]["word_accuracy"] or 0) * r["speakers"]["words"] for r in multi)

    # judge
    verdicts = [v for r in rows for v in (r.get("judge") or {}).values() if "standalone" in v]

    def judged(group: str, subset: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        pool = (
            [v for r in subset for v in (r.get("judge") or {}).values() if "standalone" in v and v["group"] == group]
            if subset is not None
            else [v for v in verdicts if v["group"] == group]
        )
        return {
            "n": len(pool),
            "standalone_pct": _pct(sum(1 for v in pool if v["standalone"]), len(pool)),
            "complete_ending_pct": _pct(sum(1 for v in pool if v["complete_ending"]), len(pool)),
        }

    # cuts
    cuts = [c[k] for c in clips for k in ("cut_in", "cut_out") if c.get(k)]
    loud = [x for x in cuts if x.get("level_db") is not None]
    rendered = [c for c in clips if c.get("render")]
    boundary = [c["boundary"] for c in rendered if c.get("boundary")]
    framing = [c["framing"] for c in rendered if c.get("framing")]
    issue_counts = Counter(code for c in clips for code in c["issues"])
    highlights = [r["highlights"] for r in synth if r.get("highlights")]
    asr_seconds = sum(r["asr_seconds"] for r in rows)
    render_s = sum(c["render"]["seconds"] for c in rendered)
    render_dur = sum(c["duration"] for c in rendered)
    llm_seconds = sum(r["stages"][s]["seconds"] for r in rows for s in ("clips", "chapters", "posts"))
    return {
        "run": name,
        **meta,
        "items": len(rows),
        "synthetic_items": len(synth),
        "real_items": len(real),
        "total_duration_s": round(duration, 1),
        "wer": {"synthetic": _pooled_wer(synth), "real": _pooled_wer(real), "all": _pooled_wer(rows)},
        "asr_realtime_factor": round(asr_seconds / duration, 4) if duration else None,
        "speakers": {
            "two_speaker_items": len(multi),
            "word_accuracy": round(spk_ok / spk_words, 4) if spk_words else None,
            "count_correct": sum(1 for r in synth if r["speakers"]["count_correct"]),
            "single_speaker_items_labelled_single": sum(1 for r in single if r["speakers"]["predicted_speakers"] == 1),
            "single_speaker_items": len(single),
        },
        "chapters": {"tol15": chap("tol15"), "tol30": chap("tol30"), "items": len(synth)},
        "clips": {
            "count": len(clips),
            "per_item_median": median(len(r["clips"]) for r in rows),
            "length_ok_pct": _pct(sum(1 for c in clips if 19.5 <= c["duration"] <= 60.5), len(clips)),
            "median_length_s": round(median(c["duration"] for c in clips), 1) if clips else None,
            "issues": dict(issue_counts),
            "with_warnings_pct": _pct(sum(1 for c in clips if c["issues"]), len(clips)),
        },
        "cuts": {
            "count": len(cuts),
            "inside_word": sum(1 for x in cuts if x["inside_word"]),
            "tight_lt_30ms": sum(
                1 for x in cuts if not x["inside_word"] and min(x["gap_before"], x["gap_after"]) < 0.03
            ),
            "loud_gt_minus12db": sum(1 for x in loud if x["level_db"] > -12),
            "median_level_db": round(median(x["level_db"] for x in loud), 1) if loud else None,
            "rendered_clips_checked": len(boundary),
            "first_word_intact_pct": _pct(sum(1 for b in boundary if b["first_ok"]), len(boundary)),
            "last_word_intact_pct": _pct(sum(1 for b in boundary if b["last_ok"]), len(boundary)),
        },
        "judge": {
            "model_clips": judged("model"),
            "model_clips_synthetic": judged("model", synth),
            "model_clips_real": judged("model", real),
            "random_windows": judged("random"),
            "planted_highlights": judged("gold"),
            "unanswered": sum(1 for r in rows for v in (r.get("judge") or {}).values() if "standalone" not in v),
        },
        "highlights": {
            "planted": sum(h["highlights"] for h in highlights),
            "found": sum(h["found"] for h in highlights),
            "recall": round(sum(h["found"] for h in highlights) / max(1, sum(h["highlights"] for h in highlights)), 3),
            "clips_mostly_on_a_highlight": sum(h["clips_mostly_on_a_highlight"] for h in highlights),
            "clips_on_synthetic": sum(len(r["clips"]) for r in synth),
        },
        "framing": {
            "tracked_clips": len(framing),
            "face_in_window_tracked_pct": _pct(
                sum(f["tracked"] * f["samples_with_face"] for f in framing),
                sum(f["samples_with_face"] for f in framing),
            ),
            "face_in_window_center_pct": _pct(
                sum(f["center"] * f["samples_with_face"] for f in framing), sum(f["samples_with_face"] for f in framing)
            ),
            "layouts": dict(Counter(c["render"]["layout"] for c in rendered)),
        },
        "timing": {
            "asr_seconds": round(asr_seconds, 1),
            "asr_seconds_per_source_minute": round(asr_seconds / duration * 60, 2) if duration else None,
            "llm_stage_seconds": round(llm_seconds, 1),
            "llm_seconds_per_source_minute": round(llm_seconds / duration * 60, 2) if duration else None,
            "render_seconds": round(render_s, 1),
            "render_seconds_per_output_minute": round(render_s / render_dur * 60, 1) if render_dur else None,
            "rendered_minutes": round(render_dur / 60, 1),
        },
        "posts": {
            "items": sum(1 for r in rows if r.get("posts")),
            "social_posts": sum(r["posts"]["social_posts"] for r in rows if r.get("posts")),
            "within_limit": sum(r["posts"]["within_limit"] for r in rows if r.get("posts")),
            "truncated_by_code": sum(r["posts"]["truncated"] for r in rows if r.get("posts")),
            "grounding_flags_social": sum(r["posts"]["flags_social"] for r in rows if r.get("posts")),
            "grounding_flags_long_form": sum(r["posts"]["flags_long_form"] for r in rows if r.get("posts")),
            "chapters_youtube_ready": sum(1 for r in rows if r.get("posts", {}).get("chapters_youtube_ready")),
        },
        "calls": ledger_stats(f"eval-{name}:"),
    }


def summarize(names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"runs": {}}
    for name in names:
        summary = summarize_run(name)
        if summary:
            out["runs"][name] = summary
    rubric = RESULTS / "rubric.json"
    if rubric.exists():
        out["rubric"] = json.loads(rubric.read_text(encoding="utf-8")).get("summary")
    smoke = RESULTS / "smoke.json"
    if smoke.exists():
        out["smoke"] = json.loads(smoke.read_text(encoding="utf-8"))
    out["all_calls"] = ledger_stats("")
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(out["runs"].items()), indent=1)[:6000])
    return out


__all__ = ["Path", "summarize", "summarize_run"]


RUBRIC = """Rubric (each 0, 1 or 2): standalone (understandable with no context), hook (first sentence makes you keep
watching), payoff (ends on a point, punchline or answer), clean_boundaries (does not start or end mid-thought),
platform_fit (one idea, 20-60 s, worth posting). A clip "passes" with standalone = 2 and no zero anywhere."""


def rubric_sample(name: str, n: int = 24, seed: int = 11) -> Path:
    """A seeded sample of clips (stratified: synthetic and real) for the human-style rubric, as Markdown."""
    import random

    rows = _items(name)
    rng = random.Random(seed)
    pool = [(r["id"], r["kind"], c) for r in rows for c in r["clips"]]
    synth = [p for p in pool if p[1] == "synthetic"]
    real = [p for p in pool if p[1] == "nasa"]
    pick = rng.sample(synth, min(len(synth), n // 2)) + rng.sample(real, min(len(real), n - n // 2))
    lines = [f"# Rubric sample: run `{name}`, {len(pick)} clips (seed {seed})", "", RUBRIC, ""]
    for item_id, kind, clip in pick:
        lines += [
            f"## {item_id} / {clip['id']} ({kind}, {clip['duration']:.0f} s): {clip['title']}",
            "",
            clip["text"],
            "",
        ]
    path = RESULTS / name / "rubric_sample.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
