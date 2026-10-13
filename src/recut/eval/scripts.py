"""Drafting the synthetic test talks: a free model expands each outline in data/testset/synthetic/outlines.json into
a spoken script with one planted, self-contained highlight per chapter. The drafts are validated (speakers, length,
exactly one highlight per chapter) and committed, so the test set does not change between runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recut.config import REPO_ROOT, Settings
from recut.eval.synth import split_sentences
from recut.llm.base import LLMError
from recut.llm.factory import model_for
from recut.llm.jsonparse import JSONReplyError, parse_json_object
from recut.text import word_count

SYNTH_DIR = REPO_ROOT / "data" / "testset" / "synthetic"

SYSTEM = """RECUT_TASK: draft_script
You write realistic scripts for spoken videos (a podcast episode or a solo talk) that will be read aloud by
text-to-speech voices. Write natural spoken English: contractions, short sentences mixed with longer ones,
rhetorical questions, the odd "you know" or "honestly", no headings or stage directions, no emoji, numbers as
digits ("$5,000", "20 hours", "70%").

Structure: exactly the chapters given, in order. Each chapter has 4-7 speaker turns and 230-330 words in total.
For a podcast, the host asks and reacts in short turns, the guest gives the longer answers; for a talk, one speaker
has 3-5 turns per chapter (paragraphs).

In every chapter exactly ONE turn is the highlight ("highlight": true), built around the highlight idea given:
- 90-140 words, spoken by the guest (podcast) or the speaker (talk);
- completely self-contained: it must make sense to someone who has heard nothing else - no "as I said", no
  "that" or "it" pointing to an earlier turn, and it names what it is talking about;
- the first sentence is a hook (a bold claim, a number, a question or the start of a story);
- the last sentence is the payoff (the lesson, the punchline, the answer).
All other turns are connective tissue: they refer back to earlier turns ("so that", "right, and", "like you
said"), transition, ask questions or add detail; they should NOT work as standalone clips.

Answer with JSON only:
{"chapters": [{"title": "<the chapter title>", "bullets": ["<2-3 short slide bullets>"], "turns":
[{"speaker": "S1", "text": "...", "highlight": false}]}]}"""


def _prompt(outline: dict[str, Any]) -> str:
    speakers = "; ".join(f"{k} = {v['name']}" for k, v in outline["speakers"].items())
    lines = [
        f"Title: {outline['title']}",
        f"Show: {outline['show']}",
        f"Format: {outline['format']}",
        f"Speakers: {speakers}",
        "Chapters:",
    ]
    for i, ch in enumerate(outline["chapters"], start=1):
        lines.append(f"{i}. {ch['title']} - about: {ch['gist']}. Highlight idea: {ch['highlight']}.")
    return "\n".join(lines)


def validate(script: dict[str, Any], outline: dict[str, Any]) -> list[str]:
    problems = []
    chapters = script.get("chapters") or []
    if len(chapters) != len(outline["chapters"]):
        problems.append(f"{len(chapters)} chapters instead of {len(outline['chapters'])}")
    for i, ch in enumerate(chapters):
        turns = ch.get("turns") or []
        highlights = [t for t in turns if t.get("highlight")]
        if len(highlights) != 1:
            problems.append(f"chapter {i + 1}: {len(highlights)} highlights")
        for t in turns:
            if t.get("speaker") not in outline["speakers"]:
                problems.append(f"chapter {i + 1}: unknown speaker {t.get('speaker')!r}")
            if not str(t.get("text") or "").strip():
                problems.append(f"chapter {i + 1}: empty turn")
        words = sum(word_count(str(t.get("text") or "")) for t in turns)
        if words < 150:
            problems.append(f"chapter {i + 1}: only {words} words")
        for h in highlights:
            hw = word_count(str(h.get("text") or ""))
            if not 60 <= hw <= 190:
                problems.append(f"chapter {i + 1}: highlight has {hw} words")
            if len(split_sentences(str(h.get("text") or ""))) < 3:
                problems.append(f"chapter {i + 1}: highlight has fewer than 3 sentences")
    return problems


def draft(outline: dict[str, Any], settings: Settings, *, attempts: int = 3) -> dict[str, Any]:
    model = model_for(settings, tag="testset_script")
    messages: list[Any] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _prompt(outline)}]
    last: list[str] = []
    for _ in range(attempts):
        try:
            completion = model.complete(messages, max_tokens=max(settings.llm_max_tokens, 32000), temperature=0.7)
            reply = parse_json_object(completion.text)
        except (LLMError, JSONReplyError) as exc:
            last = [str(exc)[:200]]
            continue
        problems = validate(reply, outline)
        if not problems:
            script = {k: outline[k] for k in ("id", "title", "show", "format", "accent", "speakers")}
            script["license"] = "CC0 1.0 (synthetic test media made for this repository)"
            script["drafted_by"] = completion.model
            script["chapters"] = [
                {**ch, "title": outline["chapters"][i]["title"]} for i, ch in enumerate(reply["chapters"])
            ]
            return script
        last = problems
        messages = [
            *messages[:2],
            {"role": "assistant", "content": json.dumps(reply)[:12000]},
            {
                "role": "user",
                "content": "Fix these problems and answer with the full JSON again: " + "; ".join(problems),
            },
        ]
    raise RuntimeError(f"{outline['id']}: no valid script after {attempts} attempts: {'; '.join(last)}")


def draft_all(settings: Settings, *, only: list[str] | None = None) -> list[Path]:
    outlines = json.loads((SYNTH_DIR / "outlines.json").read_text(encoding="utf-8"))
    out = []
    for outline in outlines:
        if only and outline["id"] not in only:
            continue
        path = SYNTH_DIR / f"{outline['id']}.script.json"
        if path.exists() and not only:
            out.append(path)
            continue
        script = draft(outline, settings)
        path.write_text(json.dumps(script, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        out.append(path)
    return out
