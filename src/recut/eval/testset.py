"""The evaluation test set: synthetic talks rendered locally (exact gold data) and public-domain NASA videos with
NASA's own captions as reference transcripts. Media is not committed; data/testset/manifest.json holds the URLs,
licences and SHA-256 checksums, and `recut testset download` / `recut testset synth` rebuild it."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from recut.config import REPO_ROOT

TESTSET_DIR = REPO_ROOT / "data" / "testset"
MEDIA_DIR = TESTSET_DIR / "media"
MANIFEST = TESTSET_DIR / "manifest.json"
SYNTH_DIR = TESTSET_DIR / "synthetic"


def load_manifest() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(MANIFEST.read_text(encoding="utf-8"))["items"]
    return data


def save_manifest(items: list[dict[str, Any]]) -> None:
    payload = {
        "about": "Recut evaluation test set. Media files live in data/testset/media/ (gitignored); "
        "`recut testset download` fetches the NASA items, `recut testset synth` renders the synthetic ones.",
        "items": items,
    }
    MANIFEST.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def media_path(item: dict[str, Any]) -> Path:
    return MEDIA_DIR / f"{item['id']}.mp4"


def captions_path(item: dict[str, Any]) -> Path:
    return MEDIA_DIR / f"{item['id']}.srt"


def gold_path(item: dict[str, Any]) -> Path:
    return MEDIA_DIR / f"{item['id']}.gold.json"


def _fetch(url: str, target: Path, expected: str | None) -> dict[str, Any]:
    if target.exists() and expected and sha256(target) == expected:
        return {"file": target.name, "status": "present"}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "recut-testset/0.1 (+https://github.com/gelevanog/ai-video-repurposer)"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        while block := response.read(1 << 20):
            handle.write(block)
    digest = sha256(tmp)
    if expected and digest != expected:
        tmp.unlink()
        return {"file": target.name, "status": "checksum mismatch", "got": digest}
    tmp.replace(target)
    return {"file": target.name, "status": "downloaded", "sha256": digest}


def download() -> list[dict[str, Any]]:
    rows = []
    for item in load_manifest():
        if item["kind"] != "nasa":
            continue
        rows.append(_fetch(item["url"], media_path(item), item.get("sha256")))
        rows.append(_fetch(item["captions_url"], captions_path(item), item.get("captions_sha256")))
    return rows


def synthesize(only: list[str] | None = None) -> list[dict[str, Any]]:
    from recut.eval.synth import build
    from recut.eval.tts import Voice

    voice = Voice()
    rows = []
    for item in load_manifest():
        if item["kind"] != "synthetic" or (only and item["id"] not in only):
            continue
        target = media_path(item)
        if target.exists() and gold_path(item).exists() and not only:
            rows.append({"id": item["id"], "status": "present"})
            continue
        script_file = REPO_ROOT / item["script"]
        script = json.loads(script_file.read_text(encoding="utf-8"))
        video, gold = build(script, MEDIA_DIR, voice=voice)
        rows.append(
            {"id": item["id"], "status": "rendered", "duration": round(gold.duration, 1), "sha256": sha256(video)}
        )
    return rows


def verify() -> list[dict[str, Any]]:
    bad = []
    for item in load_manifest():
        path = media_path(item)
        if not path.exists():
            bad.append({"id": item["id"], "problem": "missing"})
        elif item["kind"] == "nasa" and item.get("sha256") and sha256(path) != item["sha256"]:
            bad.append({"id": item["id"], "problem": "checksum mismatch"})
    return bad


_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    """Cues as (start, end, text); speaker-change marks (>>) and markup removed."""
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("﻿", "").replace("\r", "")):
        lines = [x for x in block.strip().splitlines() if x.strip()]
        timing = next((i for i, x in enumerate(lines) if _SRT_TIME.search(x)), None)
        if timing is None:
            continue
        m = _SRT_TIME.search(lines[timing])
        assert m is not None
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        body = " ".join(lines[timing + 1 :])
        body = re.sub(r"<[^>]+>", "", body).replace(">>", " ").strip()
        if body:
            cues.append((start, end, body))
    return cues


def reference_text(item: dict[str, Any], start: float = 0.0, end: float | None = None) -> str:
    """The reference transcript of an item (optionally a time range): the synthetic script or NASA's captions."""
    if item["kind"] == "synthetic":
        gold = json.loads(gold_path(item).read_text(encoding="utf-8"))
        return " ".join(s["text"] for s in gold["sentences"] if s["end"] > start and (end is None or s["start"] < end))
    cues = parse_srt(captions_path(item).read_text(encoding="utf-8", errors="replace"))
    return " ".join(c[2] for c in cues if c[1] > start and (end is None or c[0] < end))
