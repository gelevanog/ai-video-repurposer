"""A scripted chat model for tests: replies are queued per task (or one reply for every call)."""

from __future__ import annotations

from collections.abc import Sequence

from recut.llm.base import Completion, LLMError, Message
from recut.llm.fake import task_of


class ScriptedModel:
    def __init__(self, replies: dict[str, list[str] | str] | None = None, *, fail: bool = False) -> None:
        self.replies = replies or {}
        self.fail = fail
        self.calls: list[tuple[str, list[Message]]] = []

    @property
    def label(self) -> str:
        return "test/scripted"

    @property
    def is_local(self) -> bool:
        return False

    def complete(self, messages: Sequence[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion:
        task = task_of(messages)
        self.calls.append((task, list(messages)))
        if self.fail:
            raise LLMError("upstream 500: boom")
        reply = self.replies.get(task, "{}")
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        return Completion(text=reply, model="test/scripted")
