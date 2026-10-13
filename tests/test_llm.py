from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from recut.config import Settings
from recut.llm.base import BudgetExceededError, Completion, LLMError, PolicyViolationError, RetryableLLMError
from recut.llm.budget import BudgetedModel, CallLedger, DiskCache
from recut.llm.factory import build_chat_model, model_for
from recut.llm.fake import FakeModel, task_of
from recut.llm.guards import ensure_free_body, ensure_free_models, ensure_served_free
from recut.llm.jsonparse import JSONReplyError, parse_json_object
from recut.llm.openai_compat import OpenAICompatibleModel

KEY = "test-" + "key"  # built at runtime so no scanner mistakes it for a credential


def test_free_only_guard_refuses_paid_ids() -> None:
    ensure_free_models(["nvidia/nemotron-3-super-120b-a12b:free"])
    with pytest.raises(PolicyViolationError):
        ensure_free_models(["anthropic/claude-sonnet-5"])
    with pytest.raises(PolicyViolationError):
        ensure_free_models(["some/model:free:online"])


def test_free_only_guard_checks_the_request_body_and_the_served_model() -> None:
    with pytest.raises(PolicyViolationError):
        ensure_free_body({"model": "a:free", "models": ["a:free", "b/paid"]})
    with pytest.raises(PolicyViolationError):
        ensure_free_body({"model": "a:free", "plugins": [{"id": "web"}]})
    with pytest.raises(PolicyViolationError):
        ensure_served_free("openai/gpt-5")
    ensure_served_free(None)


def test_factory_refuses_a_paid_model_before_building_anything(settings: Settings) -> None:
    s = settings.model_copy(
        update={"llm_provider": "openrouter", "llm_model": "openai/gpt-5", "openrouter_api_key": None}
    )
    with pytest.raises(PolicyViolationError):
        build_chat_model(s)


def test_paid_models_need_the_explicit_opt_in(settings: Settings) -> None:
    s = settings.model_copy(
        update={
            "llm_provider": "openrouter",
            "llm_model": "openai/gpt-5",
            "openrouter_api_key": KEY,
            "allow_paid_models": True,
        }
    )
    model = build_chat_model(s)
    assert isinstance(model, OpenAICompatibleModel) and not model.require_free


def test_fallbacks_are_guarded_too(settings: Settings) -> None:
    s = settings.model_copy(
        update={
            "llm_provider": "openrouter",
            "openrouter_api_key": KEY,
            "llm_model": "a/b:free",
            "llm_fallback_models": ["c/d"],
        }
    )
    with pytest.raises(PolicyViolationError):
        build_chat_model(s)


def test_default_openrouter_model_is_free_with_free_fallbacks(settings: Settings) -> None:
    s = settings.model_copy(update={"llm_provider": "openrouter", "openrouter_api_key": KEY})
    model = build_chat_model(s)
    assert isinstance(model, OpenAICompatibleModel)
    assert model.model.endswith(":free") and all(m.endswith(":free") for m in model.fallback_models)


def test_compat_provider_is_guarded_unless_opted_in(settings: Settings) -> None:
    s = settings.model_copy(
        update={"llm_provider": "compat", "llm_model": "gpt-5-mini", "llm_base_url": "http://x/v1", "llm_api_key": KEY}
    )
    with pytest.raises(PolicyViolationError):
        build_chat_model(s)
    model = build_chat_model(s.model_copy(update={"allow_paid_models": True}))
    assert model.label == "compat/gpt-5-mini"


def test_fake_model_dispatches_on_the_task_line(settings: Settings) -> None:
    model = build_chat_model(settings)
    assert isinstance(model, FakeModel) and model.is_local
    with pytest.raises(LLMError):
        model.complete([{"role": "system", "content": "RECUT_TASK: nope"}], max_tokens=10)
    assert task_of([{"role": "system", "content": "x\nRECUT_TASK: chapters\n"}]) == "chapters"


def _transport(responses: list[httpx.Response], seen: list[dict]) -> httpx.MockTransport:  # type: ignore[type-arg]
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return responses.pop(0)

    return httpx.MockTransport(handler)


def _ok(text: str, model: str = "a/b:free") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


def _model(responses: list[httpx.Response], seen: list[dict], **kw: object) -> OpenAICompatibleModel:  # type: ignore[type-arg]
    return OpenAICompatibleModel(
        kind="openrouter",
        base_url="https://openrouter.test/api/v1",
        model="a/b:free",
        api_key=KEY,
        fallback_models=["c/d:free"],
        reasoning_effort="low",
        transport=_transport(responses, seen),
        **kw,  # type: ignore[arg-type]
    )


def test_openrouter_request_shape_and_usage() -> None:
    seen: list[dict] = []  # type: ignore[type-arg]
    completion = _model([_ok("hello")], seen).complete([{"role": "user", "content": "hi"}], max_tokens=50)
    assert completion.text == "hello" and completion.input_tokens == 3 and completion.output_tokens == 2
    assert seen[0]["models"] == ["a/b:free", "c/d:free"] and seen[0]["reasoning"] == {"effort": "low"}


def test_a_paid_served_model_is_rejected() -> None:
    with pytest.raises(PolicyViolationError):
        _model([_ok("hi", model="openai/gpt-5")], []).complete([{"role": "user", "content": "x"}], max_tokens=5)


def test_rate_limits_and_server_errors_are_retryable() -> None:
    with pytest.raises(RetryableLLMError) as info:
        _model(
            [httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}})], []
        ).complete([{"role": "user", "content": "x"}], max_tokens=5)
    assert info.value.retry_after == 7.0
    with pytest.raises(RetryableLLMError):
        _model([httpx.Response(503, json={"error": {"message": "busy"}})], []).complete(
            [{"role": "user", "content": "x"}], max_tokens=5
        )
    with pytest.raises(RetryableLLMError):  # OpenRouter reports some upstream failures inside a 200
        _model([httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}})], []).complete(
            [{"role": "user", "content": "x"}], max_tokens=5
        )


def test_a_cut_off_reply_is_an_error() -> None:
    cut = httpx.Response(
        200, json={"model": "a/b:free", "choices": [{"message": {"content": '{"cl'}, "finish_reason": "length"}]}
    )
    with pytest.raises(LLMError, match="cut off"):
        _model([cut], []).complete([{"role": "user", "content": "x"}], max_tokens=5)


def test_missing_key_is_reported() -> None:
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        OpenAICompatibleModel(kind="openrouter", base_url="https://x", model="a:free", api_key=None)


class _Flaky:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    label = "openrouter/flaky:free"
    is_local = False

    def complete(self, messages, *, max_tokens, temperature=0.0):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls <= self.failures:
            raise RetryableLLMError("rate limited", retry_after=0.01)
        return Completion(text="ok", model="flaky:free", input_tokens=1, output_tokens=1)


def test_budgeted_model_retries_records_and_caches(tmp_path: Path) -> None:
    ledger = CallLedger(tmp_path / "calls.jsonl", max_calls=10)
    inner = _Flaky(failures=2)
    model = BudgetedModel(
        inner, ledger=ledger, cache=DiskCache(tmp_path / "cache"), throttle=None, retry_base_seconds=0.01, tag="t"
    )
    messages = [{"role": "user", "content": "hi"}]
    assert model.complete(messages, max_tokens=5).text == "ok"  # type: ignore[arg-type]
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["retryable_error", "retryable_error", "ok"]
    assert all("hi" not in json.dumps(r) for r in rows)  # prompts never reach the ledger
    again = model.complete(messages, max_tokens=5)  # type: ignore[arg-type]
    assert again.cached and inner.calls == 3


def test_call_budget_is_a_hard_ceiling(tmp_path: Path) -> None:
    ledger = CallLedger(tmp_path / "calls.jsonl", max_calls=1)
    model = BudgetedModel(_Flaky(0), ledger=ledger, cache=None, throttle=None, tag="t")
    model.complete([{"role": "user", "content": "a"}], max_tokens=5)  # type: ignore[list-item]
    with pytest.raises(BudgetExceededError):
        model.complete([{"role": "user", "content": "b"}], max_tokens=5)  # type: ignore[list-item]


def test_model_for_passes_the_fake_model_through(settings: Settings) -> None:
    assert isinstance(model_for(settings, tag="x"), FakeModel)


def test_json_parsing_of_messy_replies() -> None:
    assert parse_json_object('<think>hmm {"no": 1}</think> Sure! ```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert parse_json_object('prefix {"t": “quoted”} suffix') == {"t": "quoted"}
    with pytest.raises(JSONReplyError):
        parse_json_object("no json at all")
