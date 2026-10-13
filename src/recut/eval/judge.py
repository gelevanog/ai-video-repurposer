"""The clip judge: a different free model reads only the clip's words (blind: model picks, random windows and the
planted highlights are shuffled together under neutral ids) and says whether the clip stands on its own."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from recut.config import DEFAULT_JUDGE_FALLBACKS, Settings
from recut.llm.base import LLMError
from recut.llm.factory import model_for
from recut.llm.jsonparse import JSONReplyError, parse_json_object

SYSTEM = """RECUT_TASK: judge_clips
You review short vertical video clips cut from long videos, before they are posted on TikTok, Reels and Shorts.
For each clip you get ONLY its spoken words. You have not seen the rest of the video - judge exactly as a viewer
scrolling past would.

For each clip decide:
- "standalone": true if a viewer who sees only this clip understands what is being talked about and gets a
  complete point, story or answer from it. false if it depends on context outside the clip: an unexplained "he",
  "that", "this idea" or name; "as I said"; it starts mid-thought; or it stops before the point is made.
- "complete_ending": true if the last sentence lands the point; false if it stops mid-thought or on an open question.
- "hook": 1-10, how strongly the first sentence makes a viewer keep watching.
- "payoff": 1-10, how satisfying the end is.
Be strict and consistent; most random excerpts are not standalone.
Answer with JSON only: {"clips": [{"id": "<id>", "standalone": true, "complete_ending": true, "hook": 5,
"payoff": 5, "reason": "<max 20 words>"}]}"""


@dataclass
class JudgeItem:
    id: str
    group: str
    """"model", "random" or "gold"."""
    text: str


def judge(
    items: Sequence[JudgeItem], settings: Settings, *, tag: str, seed: int = 7, cloud: bool | None = None
) -> dict[str, dict[str, Any]]:
    """Verdicts by item id. Items are shuffled and relabelled so the judge cannot tell the groups apart."""
    if not items:
        return {}
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    alias = {f"c{i + 1:02d}": item for i, item in enumerate(shuffled)}
    body = "\n\n".join(f"### Clip {key}\n{item.text}" for key, item in alias.items())
    use_cloud = settings.llm_provider != "fake" if cloud is None else cloud
    if not use_cloud:
        model = model_for(settings.model_copy(update={"llm_provider": "fake"}), tag=tag)
    else:
        model = model_for(
            settings,
            tag=tag,
            provider="openrouter",
            model=settings.judge_model,
            fallback_models=DEFAULT_JUDGE_FALLBACKS,
        )
    try:
        completion = model.complete(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}],
            max_tokens=settings.llm_max_tokens,
            temperature=0.0,
        )
        reply = parse_json_object(completion.text)
    except (LLMError, JSONReplyError) as exc:
        return {item.id: {"group": item.group, "error": str(exc)[:200]} for item in items}
    out: dict[str, dict[str, Any]] = {}
    for verdict in reply.get("clips") or []:
        if not isinstance(verdict, dict):
            continue
        key = str(verdict.get("id") or "")
        if key in alias:
            item = alias[key]
            out[item.id] = {
                "group": item.group,
                "standalone": bool(verdict.get("standalone")),
                "complete_ending": bool(verdict.get("complete_ending")),
                "hook": verdict.get("hook"),
                "payoff": verdict.get("payoff"),
                "reason": str(verdict.get("reason") or "")[:200],
                "judge": completion.model,
            }
    for item in items:
        out.setdefault(item.id, {"group": item.group, "error": "no verdict"})
    return out
