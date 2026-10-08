"""Dynamic cost tracking for agent harnesses.

Dollar costs are **no longer statically configured** in ``config/models.yaml``
— static prices cannot follow OpenRouter's provider-dependent, peak-usage and
promotion-dependent billing. Instead, costs are tracked dynamically, in
priority order:

1. **Per-response reported cost** (primary). OpenRouter includes the actual
   billed cost of every request in the LLM API response body under
   ``usage.cost`` (the chat-completions endpoint and the Anthropic-compatible
   gateway both return it; litellm additionally stashes it on the response's
   hidden params). It is computed by OpenRouter from the provider that actually
   served the request, so it reflects provider routing, peak pricing and
   promotions exactly. mini-swe-agent persists the full response on each
   message's ``extra["response"]``, and OpenHands' telemetry already prefers
   the hidden-params copy, so both litellm-based harnesses can read it.

2. **One-time OpenRouter pricing query** (fallback). Harnesses that never see
   the LLM API response body (``codex exec`` / ``claude -p`` report token
   counts only) price their token usage with per-token rates fetched **once per
   process** from OpenRouter's models/endpoints API
   (``GET /api/v1/models/<model_id>/endpoints``) for the participant's
   (model, provider) selection. The result is cached — harnesses are
   constructed per agent run, so the query effectively happens once at agent
   initialization and is reused throughout the session. It is also used as the
   last-resort fallback by the litellm harnesses when a response carries no
   reported cost.

3. **Unavailable** — the cost is ``0.0`` and a warning is emitted. The
   selection TUIs (``scripts/generate_challenges.py``, ``run_tournament*.py``)
   warn up front when a selected participant's (model, provider) has no
   pricing available from the OpenRouter API.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from swe_duel.config import ModelConfig

# The model id contains a "/" (e.g. "z-ai/glm-5.3-flash") and the endpoints API
# expects it raw in the path — percent-encoding the slash yields a 404.
_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"
_HTTP_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class TokenPricing:
    """Per-token USD prices for one (model, provider) per OpenRouter's API."""

    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_cost_per_token: float = 0.0
    # OpenRouter endpoint tag the prices came from ("" when unknown).
    source_endpoint: str = ""

    def token_cost(
        self, input_tokens: int, output_tokens: int, cached_tokens: int = 0
    ) -> float:
        """Estimate the cost of one call. Cached input bills at the cheaper
        cache-read rate; providers that do not publish one pay full input."""
        cached = max(0, min(int(input_tokens), int(cached_tokens)))
        uncached = int(input_tokens) - cached
        return (
            uncached * self.input_cost_per_token
            + cached * self.cache_read_cost_per_token
            + int(output_tokens) * self.output_cost_per_token
        )


# ── per-response reported cost ───────────────────────────────────────────


def response_reported_cost(response: Any) -> float | None:
    """The dollar cost reported in the LLM API response body, or None.

    OpenRouter returns the request's actual billed cost under
    ``usage.cost``. Accepts a serialized response dict (litellm
    ``response.model_dump()`` — what harnesses persist on message extras) or
    a live response object with a ``usage`` attribute. Returns ``None`` when
    the response carries no cost so callers can fall through to pricing.
    """
    if response is None:
        return None
    usage: Any = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
        if usage is not None and not isinstance(usage, dict):
            try:
                usage = usage.model_dump()  # pydantic (litellm Usage)
            except Exception:
                usage = None
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    return float(cost)


# ── one-time (cached) OpenRouter pricing query ────────────────────────────

_PRICING_CACHE: dict[tuple[str, str], TokenPricing | None] = {}
_PRICING_LOCK = threading.Lock()
_MISSING_PRICING_WARNED: set[tuple[str, str]] = set()


def _api_key() -> str:
    key = os.environ.get("SWE_DUEL_OPENROUTER_API_KEY") or os.environ.get(
        "OPENROUTER_API_KEY"
    )
    if not key:
        raise RuntimeError(
            "SWE_DUEL_OPENROUTER_API_KEY (or OPENROUTER_API_KEY) is required "
            "to query OpenRouter pricing"
        )
    return key


def _endpoint_matches_provider(tag: str, provider: str) -> bool:
    """Endpoint tags are provider slugs, sometimes with a quantization suffix
    (``"z-ai/fp8"``); a pinned slug like ``"z-ai"`` matches either form."""
    return tag == provider or tag.split("/", 1)[0] == provider


def _pricing_from_endpoints(
    endpoints: list[dict[str, Any]], provider: str
) -> TokenPricing | None:
    """Pick pricing from a model's OpenRouter endpoints for a provider slug.

    With a pinned provider only that provider's endpoints are considered. The
    endpoint that will actually serve a request is not knowable up front
    (auto-routing, quantization variants), so the cheapest priced endpoint is
    taken as a conservative estimate. Returns None when no candidate endpoint
    publishes both prompt and completion prices.
    """
    priced: list[TokenPricing] = []
    for ep in endpoints:
        if not isinstance(ep, dict):
            continue
        if provider and not _endpoint_matches_provider(
            str(ep.get("tag") or ""), provider
        ):
            continue
        pricing = ep.get("pricing")
        if not isinstance(pricing, dict):
            continue
        try:
            prompt = float(pricing["prompt"])
            completion = float(pricing["completion"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            cache_read = float(pricing.get("input_cache_read") or 0.0)
        except (TypeError, ValueError):
            cache_read = 0.0
        priced.append(
            TokenPricing(
                input_cost_per_token=prompt,
                output_cost_per_token=completion,
                cache_read_cost_per_token=cache_read,
                source_endpoint=str(ep.get("tag") or ""),
            )
        )
    if not priced:
        return None
    return min(
        priced, key=lambda t: (t.input_cost_per_token, t.output_cost_per_token)
    )


def fetch_openrouter_pricing(model_id: str, provider: str = "") -> TokenPricing | None:
    """Query OpenRouter's endpoints API once per (model, provider) per process.

    Results — including negative ones (unknown model, provider not serving it,
    query failure) — are cached, so repeated harness construction across an
    agent session never re-fetches. Returns ``None`` when pricing is
    unavailable.
    """
    model_id = model_id.removeprefix("openrouter/")
    key = (model_id, provider)
    with _PRICING_LOCK:
        if key in _PRICING_CACHE:
            return _PRICING_CACHE[key]
    pricing: TokenPricing | None = None
    try:
        url = _ENDPOINTS_URL.format(model_id=model_id)
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {_api_key()}"}
        )
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read())
        endpoints = (body.get("data") or {}).get("endpoints") or []
        pricing = _pricing_from_endpoints(list(endpoints), provider)
    except Exception:
        pricing = None
    with _PRICING_LOCK:
        _PRICING_CACHE[key] = pricing
    return pricing


def pricing_for(model_config: ModelConfig) -> TokenPricing | None:
    """Pricing for a participant's (model, provider) selection (cached)."""
    provider = getattr(model_config, "provider", "") or ""
    return fetch_openrouter_pricing(model_config.model_id, provider)


def _warn_missing_pricing_once(model_config: ModelConfig) -> None:
    key = (model_config.model_id, getattr(model_config, "provider", "") or "")
    if key in _MISSING_PRICING_WARNED:
        return
    _MISSING_PRICING_WARNED.add(key)
    provider = key[1]
    label = provider or "auto-route"
    warnings.warn(
        f"No OpenRouter pricing available for {key[0]} "
        f"(provider: {label}) — cost tracking reports $0.00 for it",
        stacklevel=3,
    )


def fallback_cost_for(
    model_config: ModelConfig,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Price a token-count-only response with the cached OpenRouter rates.

    Last-resort cost source for responses without a reported cost (CLI
    harnesses never see the response body): ``0.0`` when pricing is
    unavailable (warns once per (model, provider) per process).
    """
    if input_tokens <= 0 and output_tokens <= 0:
        return 0.0
    pricing = pricing_for(model_config)
    if pricing is None:
        _warn_missing_pricing_once(model_config)
        return 0.0
    return pricing.token_cost(input_tokens, output_tokens, cached_tokens)


# ── TUI startup check ─────────────────────────────────────────────────────


def warn_missing_pricing(
    model_configs: Iterable[ModelConfig],
    emit: Callable[[str], None] = print,
) -> None:
    """Warn once per (model, provider) that OpenRouter cannot price it.

    Called by the selection TUIs after participants are chosen so operators
    see, before any agent runs, which selections would report ``$0.00`` costs.
    Also warms the pricing cache for every checked combo.
    """
    seen: set[tuple[str, str]] = set()
    for cfg in model_configs:
        model_id = cfg.model_id
        provider = getattr(cfg, "provider", "") or ""
        key = (model_id, provider)
        if key in seen:
            continue
        seen.add(key)
        if fetch_openrouter_pricing(model_id, provider) is not None:
            continue
        emit(
            f"[warn] no OpenRouter pricing for {model_id} "
            f"(provider: {provider or 'auto-route'}) — cost tracking for this "
            f"participant will report $0.00"
        )
