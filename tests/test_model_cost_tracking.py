"""Cost-tracking + identity-validation tests for every model in config/models.yaml.

A participant is the 4-tuple (model, harness, reasoning_effort, provider). This
module live-probes the two LLM-facing dimensions of that identity through the
exact same code path SWE-Duel harnesses use (litellm → OpenRouter):

* **model × provider** — one integration test per (model, provider-slug)
  combination from the model's models.yaml ``providers`` menu (plus the
  default auto-routed case). Asserts the pinned provider slug routes (no
  "No endpoints found" 404) AND that input/output tokens plus a positive
  dollar cost are recoverable from the response.
* **model × reasoning effort** — one integration call per effort string in the
  model's models.yaml ``reasoning_efforts`` menu (NOT permuted with provider —
  gateway-level acceptance of the string is what needs validating). Asserts
  OpenRouter accepts the effort option (no 400 "Invalid option").

Upstream throttling (429) and gated models (403 attestation) are skipped as
inconclusive — the request already passed parameter validation in those cases.

The unit-test variant exercises ``AgentWrapper._extract_trajectory`` with a
synthetic agent fixture so per-step token/cost extraction is verified without
touching the network.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest

from swe_duel.agents.agent_wrapper import AgentWrapper
from swe_duel.agents.harness.cost_tracking import fetch_openrouter_pricing
from swe_duel.config import ModelConfig

from conftest import load_integration_models, integration_model_id


# ── Unit: trajectory extraction handles real litellm extras ──────────────


def _make_fake_agent(messages: list[dict], cost: float = 0.0, n_calls: int = 0) -> Any:
    return SimpleNamespace(messages=messages, cost=cost, n_calls=n_calls)


def _stub_pricing(monkeypatch, input_per_token: float, output_per_token: float) -> None:
    """Pin the cached OpenRouter pricing so fallback cost math is hermetic."""
    from swe_duel.agents.harness import cost_tracking

    monkeypatch.setattr(
        cost_tracking,
        "fetch_openrouter_pricing",
        lambda model_id, provider="": cost_tracking.TokenPricing(
            input_cost_per_token=input_per_token,
            output_cost_per_token=output_per_token,
        ),
    )


def test_extract_trajectory_reads_litellm_usage(record):
    """Confirm we pull tokens out of extra['response']['usage'] (the real shape)."""
    model_cfg = ModelConfig(model_id="z-ai/glm-5.3-flash")
    wrapper = AgentWrapper.__new__(AgentWrapper)
    wrapper.model_config = model_cfg

    assistant_extra = {
        "cost": 0.000849,
        "response": {
            "usage": {
                "prompt_tokens": 584,
                "completion_tokens": 53,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        },
    }
    messages = [
        {"role": "system", "content": "boot"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "thinking", "extra": assistant_extra},
        {"role": "tool", "content": "hi\n"},
    ]
    traj = wrapper._extract_trajectory(_make_fake_agent(messages, cost=0.000849, n_calls=1), 1.23)
    record("trajectory", asdict(traj))

    assert traj.total_input_tokens == 584
    assert traj.total_output_tokens == 53
    assert traj.total_cost_usd == pytest.approx(0.000849)
    assert len(traj.steps) == 1
    step = traj.steps[0]
    assert step["input_tokens"] == 584
    assert step["output_tokens"] == 53
    assert step["cost_usd"] == pytest.approx(0.000849)


def test_extract_trajectory_prefers_reported_response_body_cost(record):
    """The OpenRouter-reported cost in the response body (``usage.cost``) is
    the dynamic, authoritative per-request cost and must win over litellm's
    own computed (static-map) cost."""
    model_cfg = ModelConfig(model_id="z-ai/glm-5.3-flash")
    wrapper = AgentWrapper.__new__(AgentWrapper)
    wrapper.model_config = model_cfg

    assistant_extra = {
        "cost": 0.999,  # litellm static-map cost — must be ignored
        "response": {
            "usage": {
                "prompt_tokens": 584,
                "completion_tokens": 53,
                "cost": 0.000123,
            }
        },
    }
    messages = [
        {"role": "assistant", "content": "thinking", "extra": assistant_extra},
    ]
    traj = wrapper._extract_trajectory(_make_fake_agent(messages, cost=0.999, n_calls=1), 1.0)
    record("trajectory", asdict(traj))

    assert traj.steps[0]["cost_usd"] == pytest.approx(0.000123)
    # The dynamic per-step sum also drives the total, not the accumulated
    # litellm static-map cost (0.999).
    assert traj.total_cost_usd == pytest.approx(0.000123)


def test_extract_trajectory_fallback_cost_when_usage_only(monkeypatch, record):
    """If no response-body cost and no litellm cost are present but token
    counts are, we price the call with the cached OpenRouter per-token rates."""
    _stub_pricing(monkeypatch, 0.002, 0.004)
    model_cfg = ModelConfig(model_id="qwen/qwen3.8-flash")
    wrapper = AgentWrapper.__new__(AgentWrapper)
    wrapper.model_config = model_cfg

    assistant_extra = {
        "cost": 0.0,
        "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
    }
    messages = [
        {"role": "assistant", "content": "thinking", "extra": assistant_extra},
        {"role": "tool", "content": "obs"},
    ]
    traj = wrapper._extract_trajectory(_make_fake_agent(messages), 0.5)
    record("trajectory", asdict(traj))

    # 100 * 0.002 + 50 * 0.004 = 0.2 + 0.2 = 0.4
    assert traj.total_cost_usd == pytest.approx(0.4)
    assert traj.steps[0]["cost_usd"] == pytest.approx(0.4)


# ── Unit: identity helpers + selection plumbing ───────────────────────────


def test_composite_id_roundtrip_with_effort_and_provider():
    from swe_duel.models import composite_id, split_composite_id

    # Default identity keeps the legacy 2-part form byte-for-byte.
    assert composite_id("z-ai/glm-5.3-flash", "mini-swe-agent") == (
        "z-ai/glm-5.3-flash#mini-swe-agent"
    )
    assert split_composite_id("z-ai/glm-5.3-flash#mini-swe-agent") == (
        "z-ai/glm-5.3-flash", "mini-swe-agent", "", "",
    )
    # Effort + provider extend the id with two further segments.
    cid = composite_id("z-ai/glm-5.3-flash", "codex", "high", "fireworks")
    assert cid == "z-ai/glm-5.3-flash#codex#high#fireworks"
    assert split_composite_id(cid) == (
        "z-ai/glm-5.3-flash", "codex", "high", "fireworks",
    )
    # Either extension alone preserves positional empty segments.
    assert split_composite_id("m#h##cloudflare") == ("m", "h", "", "cloudflare")
    assert split_composite_id("m#h#low#") == ("m", "h", "low", "")
    # Bare model ids and legacy 2-part ids stay parseable.
    assert split_composite_id("z-ai/glm-5.3-flash") == ("z-ai/glm-5.3-flash", "", "", "")


def test_model_config_selection_copy():
    base = ModelConfig(model_id="z-ai/glm-5.3-flash", reasoning_efforts=["high", "low"])
    assert base.reasoning_effort == "" and base.provider == ""
    sel = base.with_selection("high", "cloudflare")
    assert sel.reasoning_effort == "high" and sel.provider == "cloudflare"
    # Base config is untouched (per-participant copies only).
    assert base.reasoning_effort == "" and base.provider == ""
    assert sel.model_id == base.model_id


def test_model_config_rejects_identity_corrupting_selections():
    with pytest.raises(ValueError):
        ModelConfig(model_id="z-ai/glm-5.3-flash", provider="bad#slug")
    with pytest.raises(ValueError):
        ModelConfig(model_id="z-ai/glm-5.3-flash", reasoning_effort="has space")
    with pytest.raises(ValueError):
        ModelConfig(model_id="z-ai/glm-5.3-flash", providers=["dup", "dup"])


# ── Integration: live OpenRouter probes ───────────────────────────────────


def _model_provider_cases() -> list[tuple[ModelConfig, str]]:
    """(model, provider) pairs: every models.yaml menu slug + the auto-route
    default ("") — the exact competitor identities generation can run as."""
    return [
        (mc, provider)
        for mc in load_integration_models()
        for provider in ([""] + list(mc.providers))
    ]


def _case_id(mc: ModelConfig, provider: str) -> str:
    return f"{integration_model_id(mc)}__{provider or 'auto'}"


def _model_effort_cases() -> list[tuple[ModelConfig, str]]:
    """(model, effort) pairs — every effort string in the model's yaml menu."""
    return [
        (mc, effort)
        for mc in load_integration_models()
        for effort in mc.reasoning_efforts
    ]


def _effort_case_id(mc: ModelConfig, effort: str) -> str:
    return f"{integration_model_id(mc)}__effort-{effort}"


class _InconclusiveSkip(Exception):
    """The probe could not be judged (upstream throttle / gated model)."""


def _probe(model_config: ModelConfig, extra_body: dict, max_retries: int = 2):
    """One live litellm→OpenRouter call, with light 429 retry/backoff.

    Returns (usage_dict, reported_cost) — the parsed usage body (which carries
    OpenRouter's per-request ``cost``) and the cost extracted from it via the
    production :func:`response_reported_cost` helper. Raises the underlying
    litellm error so the caller can classify it; ``_InconclusiveSkip``
    escapes throttled or gated models after the retries are spent.
    """
    import litellm

    from swe_duel.agents.harness.cost_tracking import response_reported_cost

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            time.sleep(5 * attempt)
        try:
            response = litellm.completion(
                model=model_config.openrouter_model_id,
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                temperature=0.0,
                max_tokens=512,
                extra_body=extra_body,
            )
            break
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "status_code", None)
            if code == 429:
                last_exc = e
                continue
            raise
    else:
        raise _InconclusiveSkip(f"rate-limited upstream: {last_exc}") from last_exc

    response_dict = (
        response.model_dump() if hasattr(response, "model_dump") else dict(response)
    )
    usage = response_dict.get("usage") or {}
    reported = response_reported_cost(response_dict)
    return usage, (reported if reported is not None else 0.0)


def _classify_gating_error(exc: Exception) -> None:
    """Turn account/upstream gates into skips; re-raise real validation errors."""
    code = getattr(exc, "status_code", None)
    msg = str(exc)
    if code == 403 or "attestation" in msg.lower():
        raise _InconclusiveSkip(f"model gated (403 attestation): {msg[:120]}")
    if code and 500 <= int(code) < 600:
        raise _InconclusiveSkip(f"upstream {code}: {msg[:120]}")
    # Everything else (404 unknown endpoints, 400 invalid option, …) is real.
    raise exc


def _guard_clip(key: str) -> str:
    assert os.environ.get("SWE_DUEL_OPENROUTER_API_KEY"), (
        "SWE_DUEL_OPENROUTER_API_KEY not set"
    )
    return key


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_config,provider",
    _model_provider_cases(),
    ids=[_case_id(mc, p) for mc, p in _model_provider_cases()],
)
def test_model_cost_tracking_compatibility(
    model_config: ModelConfig, provider: str, record
):
    """Hit each (model, provider) competitor identity once via the real
    OpenRouter path (pinned provider routing, no fallbacks) and verify the
    response carries tokens + a positive cost (or, failing that, can be priced
    locally). A 404 "No endpoints found" fails the test: the models.yaml
    provider menu names a slug that does not serve this model."""
    _guard_clip("provider probe")
    extra_body: dict = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}

    try:
        usage, cost = _probe(model_config, extra_body)
    except _InconclusiveSkip as e:
        pytest.skip(f"inconclusive: {e}")
    except Exception as e:  # noqa: BLE001
        if getattr(e, "status_code", None) == 404:
            pytest.fail(
                f"models.yaml provider {provider!r} does not route for "
                f"{model_config.model_id} (OpenRouter: no endpoints found): {e}"
            )
        try:
            _classify_gating_error(e)
        except _InconclusiveSkip as skip:
            pytest.skip(f"inconclusive: {skip}")

    in_tok = int(usage.get("prompt_tokens", 0) or 0)
    out_tok = int(usage.get("completion_tokens", 0) or 0)
    # The two dynamic cost sources: the per-response cost OpenRouter reports in
    # the usage body, and (as a floor) the per-token pricing from the
    # OpenRouter endpoints API for this (model, provider).
    reported_cost = cost
    pricing = fetch_openrouter_pricing(model_config.model_id, provider)
    fallback_cost = (
        pricing.token_cost(in_tok, out_tok) if pricing is not None else 0.0
    )

    record(
        "probe",
        {
            "model_id": model_config.model_id,
            "openrouter_id": model_config.openrouter_model_id,
            "provider": provider or "(auto)",
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "reported_cost_usd": reported_cost,
            "fallback_cost_usd": fallback_cost,
            "has_usage_field": bool(usage),
        },
    )

    assert in_tok > 0, f"Model {model_config.model_id} reported no prompt_tokens"
    assert out_tok > 0, f"Model {model_config.model_id} reported no completion_tokens"
    # Cost must be recoverable from the response body OR via the pricing API.
    assert reported_cost > 0.0 or fallback_cost > 0.0, (
        f"Model {model_config.model_id}: OpenRouter reported no usage.cost and "
        f"the endpoints API has no pricing for provider {provider or '(auto)'}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_config,effort",
    _model_effort_cases(),
    ids=[_effort_case_id(mc, e) for mc, e in _model_effort_cases()],
)
def test_reasoning_effort_option_is_valid(
    model_config: ModelConfig, effort: str, record
):
    """Call OpenRouter once per (model, reasoning-effort) from the models.yaml
    menu and confirm the effort string is accepted. A 400 "Invalid option:
    expected one of ..." fails the test — the yaml menu names an effort this
    model/gateway rejects. Not permuted with provider: model–provider routing
    is covered by test_model_cost_tracking_compatibility."""
    _guard_clip("effort probe")
    extra_body = {"reasoning": {"effort": effort}}

    try:
        usage, cost = _probe(model_config, extra_body)
    except _InconclusiveSkip as e:
        pytest.skip(f"inconclusive: {e}")
    except Exception as e:  # noqa: BLE001
        if getattr(e, "status_code", None) == 400:
            pytest.fail(
                f"models.yaml reasoning_efforts entry {effort!r} is rejected by "
                f"OpenRouter for {model_config.model_id}: {e}"
            )
        try:
            _classify_gating_error(e)
        except _InconclusiveSkip as skip:
            pytest.skip(f"inconclusive: {skip}")

    record(
        "effort_probe",
        {
            "model_id": model_config.model_id,
            "reasoning_effort": effort,
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "reported_cost_usd": cost,
        },
    )
    assert int(usage.get("prompt_tokens", 0) or 0) > 0, (
        f"Effort {effort!r} on {model_config.model_id}: no usage reported"
    )
