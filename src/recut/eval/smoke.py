"""Smoke test of candidate free models before a real run: one short clip-picking task each, JSON validity,
latency and whether the picked range is plausible. Writes results/smoke.json."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from recut.clips.suggest import SYSTEM, parse_candidates, transcript_lines
from recut.config import Settings
from recut.llm.base import LLMError
from recut.llm.factory import model_for
from recut.llm.jsonparse import JSONReplyError, parse_json_object
from recut.models import Transcript
from recut.text import sentences_from_words


def run(models: list[str], sample: Path, settings: Settings, out: Path) -> list[dict[str, Any]]:
    transcript = Transcript.model_validate_json(sample.read_text(encoding="utf-8"))
    sentences = sentences_from_words(transcript.words)
    rows: list[dict[str, Any]] = []
    for name in models:
        model = model_for(settings, tag="smoke", provider="openrouter", model=name, fallback_models=[])
        started = time.monotonic()
        row: dict[str, Any] = {"model": name}
        try:
            completion = model.complete(
                [
                    {"role": "system", "content": SYSTEM.format(min_s=20, max_s=60, n=2)},
                    {"role": "user", "content": transcript_lines(sentences)},
                ],
                max_tokens=settings.llm_max_tokens,
                temperature=0.2,
            )
            reply = parse_json_object(completion.text)
            found, bad = parse_candidates(reply, {s.index for s in sentences})
            row.update(
                ok=bool(found),
                clips=[(c.first, c.last, c.title) for c in found],
                rejected=bad,
                served=completion.model,
                output_tokens=completion.output_tokens,
            )
        except (LLMError, JSONReplyError) as exc:
            row.update(ok=False, error=str(exc)[:300])
        row["seconds"] = round(time.monotonic() - started, 1)
        rows.append(row)
        print(json.dumps(row))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    return rows
