"""Unit tests for dynamic cost tracking (swe_duel.agents.harness.cost_tracking).

Hermetic: the OpenRouter endpoints API is monkeypatched everywhere — these
tests never touch the network.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from swe_duel.agents.harness import cost_tracking as ct
from swe_duel.config import ModelConfig


def _endpoints_body(*endpoints: dict[str, Any]) -> bytes:
    return json.dumps({"data": {"endpoints": list(endpoints)}}).encode()


def _ep(tag: str, prompt: str, completion: str, **extra: Any) -> dict[str, Any]:
    pricing: dict[str, Any] = {"prompt": prompt, "completion": completion}
    pricing.update(extra)
    return {"tag": tag, "provider_name": tag, "pricing": pricing}


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


# ── response_reported_cost ────────────────────────────────────────────────


def test_response_reported_cost_from_dict():
    resp = {"usage": {"prompt_tokens": 5, "cost": 5.5e-06}}
    assert ct.response_reported_cost(resp) == pytest.approx(5.5e-06)


def test_response_reported_cost_zero_is_reported():
    """An explicit 0.0 body cost is authoritative (free model), not 'missing'."""
    assert ct.response_reported_cost({"usage": {"cost": 0.0}}) == 0.0


def test_response_reported_cost_missing_returns_none():
    assert ct.response_reported_cost({"usage": {"prompt_tokens": 5}}) is None
    assert ct.response_reported_cost({}) is None
    assert ct.response_reported_cost(None) is None
    assert ct.response_reported_cost({"usage": {"cost": "not-a-number"}}) is None
    assert ct.response_reported_cost({"usage": {"cost": True}}) is None


def test_response_reported_cost_from_pydantic_like_usage():
    class _Usage:
        def model_dump(self) -> dict[str, Any]:
            return {"prompt_tokens": 1, "cost": 0.25}

    class _Response:
        usage = _Usage()

    assert ct.response_reported_cost(_Response()) == pytest.approx(0.25)


# ── TokenPricing.token_cost ────────────────────────────────────────────────


def test_token_pricing_math():
    p = ct.TokenPricing(input_cost_per_token=1e-6, output_cost_per_token=2e-6)
    assert p.token_cost(100, 50) == pytest.approx(100e-6 + 100e-6)


def test_token_pricing_cached_tokens():
    p = ct.TokenPricing(
        input_cost_per_token=1e-6,
        output_cost_per_token=2e-6,
        cache_read_cost_per_token=1e-7,
    )
    # 90 uncached + 10 cached inputs + 20 outputs.
    assert p.token_cost(100, 20, 10) == pytest.approx(90e-6 + 1e-6 + 40e-6)
    # Cached tokens can never exceed input tokens (clamped).
    assert p.token_cost(5, 0, 50) == pytest.approx(5 * 1e-7)


# ── endpoint selection / pricing parsing ──────────────────────────────────


def test_pricing_from_endpoints_pinned_provider_slug_matching():
    eps = [
        _ep("z-ai/fp8", "0.0000001", "0.0000003"),
        _ep("cloudflare", "0.0000002", "0.0000004"),
        _ep("deepinfra/fp8", "0.00000015", "0.00000035"),
    ]
    # Exact slug and quantization-suffixed tags both match the pinned slug.
    p = ct._pricing_from_endpoints(eps, "z-ai")
    assert p is not None and p.source_endpoint == "z-ai/fp8"
    p = ct._pricing_from_endpoints(eps, "cloudflare")
    assert p is not None and p.source_endpoint == "cloudflare"


def test_pricing_from_endpoints_auto_route_takes_cheapest():
    eps = [
        _ep("a", "0.0000002", "0.0000004"),
        _ep("b", "0.0000001", "0.0000003"),
    ]
    p = ct._pricing_from_endpoints(eps, "")
    assert p is not None and p.source_endpoint == "b"


def test_pricing_from_endpoints_unavailable():
    # No endpoints at all.
    assert ct._pricing_from_endpoints([], "z-ai") is None
    # Pinned provider not serving the model.
    assert ct._pricing_from_endpoints([_ep("a", "1", "1")], "z-ai") is None
    # Endpoints without both prompt/completion prices are skipped.
    no_price = {"tag": "a", "pricing": {"prompt": "0.1"}}
    assert ct._pricing_from_endpoints([no_price], "") is None
    # A free model (0/0) IS available pricing.
    p = ct._pricing_from_endpoints([_ep("free", "0", "0")], "")
    assert p is not None and p.token_cost(1000, 1000) == 0.0


# ── fetch_openrouter_pricing: caching + request shape ─────────────────────


@pytest.fixture(autouse=True)
def _clear_pricing_cache():
    ct._PRICING_CACHE.clear()
    ct._MISSING_PRICING_WARNED.clear()
    yield
    ct._PRICING_CACHE.clear()
    ct._MISSING_PRICING_WARNED.clear()


def test_fetch_pricing_queries_once_and_caches(monkeypatch):
    calls: list[str] = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeResponse(_endpoints_body(_ep("z-ai/fp8", "0.0000001", "0.0000003")))

    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr(ct.urllib.request, "urlopen", fake_urlopen)

    p1 = ct.fetch_openrouter_pricing("z-ai/glm-5.3-flash")
    p2 = ct.fetch_openrouter_pricing("z-ai/glm-5.3-flash")
    assert p1 is p2
    assert p1 is not None
    assert p1.input_cost_per_token == pytest.approx(1e-7)
    assert calls == ["https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints"]
    # Negative results are cached too — failures must not re-query per run.
    def failing_urlopen(request, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(ct.urllib.request, "urlopen", failing_urlopen)
    assert ct.fetch_openrouter_pricing("vendor/other-model") is None
    # The negatively-cached result must now be served WITHOUT any query.
    def must_not_query(request, timeout=None):
        raise AssertionError("negatively-cached (model, provider) re-queried")

    monkeypatch.setattr(ct.urllib.request, "urlopen", must_not_query)
    assert ct.fetch_openrouter_pricing("vendor/other-model") is None
    # Only the first (successful) model ever hit the appending fake.
    assert len(calls) == 1


def test_fetch_pricing_strips_openrouter_prefix(monkeypatch):
    seen: list[str] = []

    def fake_urlopen(request, timeout=None):
        seen.append(request.full_url)
        return _FakeResponse(_endpoints_body(_ep("z-ai", "0.0000001", "0.0000003")))

    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr(ct.urllib.request, "urlopen", fake_urlopen)
    ct.fetch_openrouter_pricing("openrouter/z-ai/some-model")
    assert seen == ["https://openrouter.ai/api/v1/models/z-ai/some-model/endpoints"]


def test_fetch_pricing_requires_api_key(monkeypatch):
    monkeypatch.delenv("SWE_DUEL_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert ct.fetch_openrouter_pricing("z-ai/glm-5.3-flash") is None


# ── fallback_cost_for ─────────────────────────────────────────────────────


def test_fallback_cost_for_zero_tokens_never_fetches(monkeypatch):
    def fail(request, timeout=None):
        raise AssertionError("must not query the pricing API for a 0-token call")

    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr(ct.urllib.request, "urlopen", fail)
    mc = ModelConfig(model_id="z-ai/glm-5.3-flash")
    assert ct.fallback_cost_for(mc, 0, 0, 0) == 0.0


def test_fallback_cost_for_warns_once_when_pricing_missing(monkeypatch):
    def fail(request, timeout=None):
        raise OSError("network down")

    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr(ct.urllib.request, "urlopen", fail)
    mc = ModelConfig(model_id="z-ai/glm-5.3-flash")

    with pytest.warns(UserWarning, match="No OpenRouter pricing available"):
        assert ct.fallback_cost_for(mc, 100, 50) == 0.0
    # Second call: pricing is negatively cached, warning fires only once.
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning becomes an error
        assert ct.fallback_cost_for(mc, 100, 50) == 0.0


def test_fallback_cost_for_prices_with_cached_pricing(monkeypatch):
    monkeypatch.setattr(
        ct,
        "fetch_openrouter_pricing",
        lambda model_id, provider="": ct.TokenPricing(
            input_cost_per_token=1e-6, output_cost_per_token=2e-6
        ),
    )
    mc = ModelConfig(model_id="z-ai/glm-5.3-flash")
    assert ct.fallback_cost_for(mc, 100, 50, 10) == pytest.approx(
        90e-6 + 100e-6
    )


# ── TUI warn helper ───────────────────────────────────────────────────────


def test_warn_missing_pricing_dedupes_and_warms_cache(monkeypatch):
    def fake_urlopen(request, timeout=None):
        # Model a is served by endpoints tagged "pa" and "x"; model b 404s.
        if "/models/vendor/a/endpoints" in request.full_url:
            return _FakeResponse(
                _endpoints_body(_ep("pa", "0.1", "0.2"), _ep("x", "0.1", "0.2"))
            )
        raise urllib.error.HTTPError(request.full_url, 404, "not found", None, None)

    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr(ct.urllib.request, "urlopen", fake_urlopen)

    emitted: list[str] = []
    cfgs = [
        ModelConfig(model_id="vendor/a"),
        ModelConfig(model_id="vendor/a", provider="x"),
        ModelConfig(model_id="vendor/a", provider="x"),  # same pair again
        ModelConfig(model_id="vendor/b"),
    ]
    ct.warn_missing_pricing(cfgs, emit=emitted.append)
    assert len(emitted) == 1
    assert "vendor/b" in emitted[0]
    assert "vendor/a" not in emitted[0]
    # Every (model, provider) pair is now cached — a second invocation reports
    # the same state again (it is a check, not a once-per-process warning) but
    # performs no further HTTP requests.
    calls: list[str] = []

    def counting_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeResponse(_endpoints_body(_ep("pa", "0.1", "0.2")))

    monkeypatch.setattr(ct.urllib.request, "urlopen", counting_urlopen)
    ct.warn_missing_pricing(cfgs, emit=emitted.append)
    assert len(emitted) == 2
    assert "vendor/b" in emitted[1]
    assert calls == []
