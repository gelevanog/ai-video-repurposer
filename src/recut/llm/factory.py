"""Builds the configured chat model, applying the free-only guard first, and wraps cloud models in the budget
wrapper (throttle, retries with backoff, hard call budget, ledger, optional disk cache)."""

from __future__ import annotations

from recut.config import DEFAULT_MODELS, LLMProviderKind, Settings
from recut.llm.base import ChatModel
from recut.llm.budget import BudgetedModel, DiskCache, shared_ledger, shared_throttle
from recut.llm.fake import FakeModel
from recut.llm.guards import ensure_free_models
from recut.llm.openai_compat import GUARDED_KINDS, OpenAICompatibleModel


def build_chat_model(
    settings: Settings,
    *,
    provider: LLMProviderKind | None = None,
    model: str | None = None,
    fallback_models: list[str] | None = None,
) -> ChatModel:
    kind: LLMProviderKind = provider or settings.llm_provider
    name = model or (settings.llm_model if provider is None else "") or DEFAULT_MODELS[kind]
    if fallback_models is not None:
        fallbacks = fallback_models
    elif model is None and provider is None:
        fallbacks = settings.fallbacks
    else:
        fallbacks = settings.llm_fallback_models
    # Guards first: nothing is constructed (no client, no key read) for a refused model.
    if kind in GUARDED_KINDS and settings.free_only:
        ensure_free_models([name, *(fallbacks if kind == "openrouter" else [])])
    if kind == "fake":
        from recut.fake_tasks import HANDLERS  # the rule-based tasks import the clip and chapter modules

        return FakeModel(HANDLERS)
    base_url = settings.openrouter_base_url if kind == "openrouter" else settings.llm_base_url
    api_key = settings.openrouter_api_key if kind == "openrouter" else settings.llm_api_key
    return OpenAICompatibleModel(
        kind=kind,
        base_url=base_url,
        model=name,
        api_key=api_key,
        fallback_models=fallbacks if kind == "openrouter" else (),
        require_free=settings.free_only,
        timeout_seconds=settings.llm_timeout_seconds,
        reasoning_effort=settings.llm_reasoning_effort if kind == "openrouter" else "",
        send_temperature=settings.llm_send_temperature,
    )


def budgeted(model: ChatModel, settings: Settings, *, tag: str, cache: bool | None = None) -> ChatModel:
    """Cloud models get the throttle, retries, call budget and ledger; the fake model passes through."""
    if model.is_local:
        return model
    use_cache = settings.llm_cache if cache is None else cache
    ledger = shared_ledger(settings.llm_ledger, settings.llm_max_calls)
    throttle = shared_throttle(settings.llm_min_seconds_between_requests)
    policy_fallback: ChatModel | None = None
    if isinstance(model, OpenAICompatibleModel) and model.kind == "openrouter" and model.fallback_models:
        # A provider-side 403 is not covered by OpenRouter's `models` list: ask the next free model directly.
        nxt, *rest = model.fallback_models
        alternative = OpenAICompatibleModel(
            kind="openrouter",
            base_url=settings.openrouter_base_url,
            model=nxt,
            api_key=settings.openrouter_api_key,
            fallback_models=rest,
            require_free=settings.free_only,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=settings.llm_reasoning_effort,
        )
        policy_fallback = BudgetedModel(
            alternative,
            ledger=ledger,
            cache=None,
            throttle=throttle,
            max_retries=settings.llm_max_retries,
            tag=f"{tag}:policy_fallback",
        )
    return BudgetedModel(
        model,
        ledger=ledger,
        cache=DiskCache(settings.llm_cache_dir) if use_cache else None,
        throttle=throttle,
        max_retries=settings.llm_max_retries,
        tag=tag,
        policy_fallback=policy_fallback,
    )


def model_for(
    settings: Settings,
    *,
    tag: str,
    provider: LLMProviderKind | None = None,
    model: str | None = None,
    fallback_models: list[str] | None = None,
) -> ChatModel:
    return budgeted(
        build_chat_model(settings, provider=provider, model=model, fallback_models=fallback_models),
        settings,
        tag=tag,
    )
