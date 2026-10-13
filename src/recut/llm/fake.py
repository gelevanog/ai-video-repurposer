"""The deterministic offline model. Every Recut prompt starts with a ``RECUT_TASK: <name>`` line; the fake model
dispatches on it to a rule-based implementation of the same task: a heuristic moment finder (hook words, questions,
numbers, sentence boundaries), lexical-cohesion chapter segmentation (TextTiling-style) and template-based posts. It
makes CI, the tests and the zero-key Docker demo work without any API key, and it is the "rules only" baseline in the
evaluation."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

from recut.llm.base import Completion, LLMError, Message

TASK_MARKER = "RECUT_TASK:"
_TASK = re.compile(r"^RECUT_TASK:\s*([a-z_]+)", re.MULTILINE)

FakeHandler = Callable[[Sequence[Message]], str]


def task_of(messages: Sequence[Message]) -> str:
    for message in messages:
        if message["role"] == "system" and (match := _TASK.search(message["content"])):
            return match.group(1)
    return ""


class FakeModel:
    def __init__(self, handlers: Mapping[str, FakeHandler]) -> None:
        self.handlers = dict(handlers)
        self.calls = 0

    @property
    def label(self) -> str:
        return "fake/rules"

    @property
    def is_local(self) -> bool:
        return True

    def complete(self, messages: Sequence[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion:
        self.calls += 1
        task = task_of(messages)
        handler = self.handlers.get(task)
        if handler is None:
            raise LLMError(f"the offline model has no handler for task {task!r}")
        return Completion(text=handler(messages), model=self.label)
