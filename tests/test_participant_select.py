"""Participant-selection helper tests (no TTY; wizard pages are questionary
prompts exercised interactively, so these cover the pure logic underneath)."""

from __future__ import annotations

import pytest

from swe_duel.config import ModelConfig
from swe_duel.models import composite_id

from swe_duel.cli.participant_select import (
    Participant,
    _effort_options,
    _provider_options,
    bank_status,
    participant_model_configs,
    print_pool_table,
    resolve_cli_participants,
)


def _models() -> dict[str, ModelConfig]:
    return {
        "glm": ModelConfig(
            model_id="z-ai/glm-5.3-flash",
            reasoning_efforts=["max", "high", "low"],
            providers=["z-ai", "cloudflare", "fireworks"],
        ),
        "muse": ModelConfig(
            model_id="meta/muse-spark-1.2-contributor",
            reasoning_efforts=["xhigh", "medium"],
            providers=["meta"],
        ),
    }


# ── option builders ─────────────────────────────────────────


def test_effort_options_filtered_by_harness_capability():
    models = _models()
    glm = models["glm"]
    # litellm-backed harnesses take any string + the model-default escape hatch.
    assert _effort_options(glm, "mini-swe-agent") == ["max", "high", "low", ""]
    # codex drops "max" (not in its config enum) but keeps the rest.
    assert _effort_options(glm, "codex") == ["high", "low", ""]
    # claude-code has no effort control at all → only "(model default)".
    assert _effort_options(glm, "claude-code") == [""]
    # A model without an effort menu is model-default only.
    assert _effort_options(ModelConfig(model_id="x/y"), "mini-swe-agent") == [""]


def test_provider_options_filtered_by_harness_capability():
    glm = _models()["glm"]
    assert _provider_options(glm, "mini-swe-agent") == [
        "z-ai", "cloudflare", "fireworks", "",
    ]
    # CLI harnesses cannot pin providers → auto-route only.
    assert _provider_options(glm, "codex") == [""]
    assert _provider_options(glm, "claude-code") == [""]
    # No menu → auto-route only.
    assert _provider_options(ModelConfig(model_id="x/y"), "openhands") == [""]


# ── Participant / CLI resolution ─────────────────────────────


def test_participant_identity_fields():
    p = Participant(
        nick="glm",
        model_config=_models()["glm"].with_selection("high", "cloudflare"),
        harness_id="codex",
        reasoning_effort="high",
        provider="cloudflare",
    )
    assert p.model_id == "z-ai/glm-5.3-flash"
    assert p.cid == "z-ai/glm-5.3-flash#codex#high#cloudflare"
    assert "effort=high" in p.label and "provider=cloudflare" in p.label
    # Default identity collapses to the legacy 2-part composite.
    d = Participant(
        nick="glm",
        model_config=_models()["glm"],
        harness_id="mini-swe-agent",
        reasoning_effort="",
        provider="",
    )
    assert d.cid == "z-ai/glm-5.3-flash#mini-swe-agent"


def test_resolve_cli_participants_cross_product():
    parts = resolve_cli_participants(
        _models(),
        models=["glm"],
        harnesses=["mini-swe-agent", "openhands"],
        efforts=["high", "low"],
        providers=["cloudflare"],
    )
    cids = [p.cid for p in parts]
    assert cids == [
        composite_id("z-ai/glm-5.3-flash", "mini-swe-agent", "high", "cloudflare"),
        composite_id("z-ai/glm-5.3-flash", "mini-swe-agent", "low", "cloudflare"),
        composite_id("z-ai/glm-5.3-flash", "openhands", "high", "cloudflare"),
        composite_id("z-ai/glm-5.3-flash", "openhands", "low", "cloudflare"),
    ]
    # Each resolved participant carries the pre-bound selection.
    for p in parts:
        assert p.model_config.reasoning_effort in ("high", "low")
        assert p.model_config.provider == "cloudflare"


def test_resolve_cli_participants_default_identity():
    parts = resolve_cli_participants(
        _models(), models=["glm"], harnesses=["mini-swe-agent"],
        efforts=None, providers=None,
    )
    assert [p.cid for p in parts] == ["z-ai/glm-5.3-flash#mini-swe-agent"]


def test_resolve_cli_participants_validates_menus_and_capabilities():
    with pytest.raises(SystemExit, match="not in the models.yaml menu"):
        resolve_cli_participants(
            _models(), models=["glm"], harnesses=["mini-swe-agent"],
            efforts=["banana"], providers=None,
        )
    with pytest.raises(SystemExit, match="not in the models.yaml menu"):
        resolve_cli_participants(
            _models(), models=["glm"], harnesses=["mini-swe-agent"],
            efforts=None, providers=["no-such-provider"],
        )
    # codex cannot express "max".
    with pytest.raises(SystemExit, match="cannot express reasoning effort"):
        resolve_cli_participants(
            _models(), models=["glm"], harnesses=["codex"],
            efforts=["max"], providers=None,
        )
    # codex cannot pin providers.
    with pytest.raises(SystemExit, match="cannot pin an OpenRouter provider"):
        resolve_cli_participants(
            _models(), models=["glm"], harnesses=["codex"],
            efforts=None, providers=["cloudflare"],
        )
    with pytest.raises(SystemExit, match="none of --models"):
        resolve_cli_participants(
            _models(), models=["nope"], harnesses=["mini-swe-agent"],
            efforts=None, providers=None,
        )


def test_participant_model_configs_keyed_by_cid():
    parts = resolve_cli_participants(
        _models(), models=["glm", "muse"], harnesses=["mini-swe-agent"],
        efforts=None, providers=None,
    )
    cfgs = participant_model_configs(parts)
    assert set(cfgs) == {
        "z-ai/glm-5.3-flash#mini-swe-agent",
        "meta/muse-spark-1.2-contributor#mini-swe-agent",
    }


# ── bank status / pool table ─────────────────────────────────


def _store_with_identity_pools(tmp_path):
    from swe_duel.challenge_bank.store import ChallengeStore
    from tests.test_challenge_store import _record

    store = ChallengeStore(bank_dir=tmp_path / "bank")
    store.store(_record(effort="", provider=""))
    store.store(_record(effort="high", provider="cloudflare"))
    store.store(_record(effort="high", provider="", harness="codex"))
    return store


def test_bank_status_groups_by_identity(tmp_path):
    store = _store_with_identity_pools(tmp_path)
    bank_ids, model_status, slot_status = bank_status(store, _models(), ["flask"])
    assert "anthropic/claude-x#mini-swe-agent" in bank_ids
    assert "anthropic/claude-x#mini-swe-agent#high#cloudflare" in bank_ids
    assert "anthropic/claude-x#codex#high#" in bank_ids
    # Per-identity landing rows; the model is not in _models() so model_status
    # stays empty (unknown models are skipped).
    assert "anthropic/claude-x#mini-swe-agent" in slot_status
    assert "w/ slots" in slot_status["anthropic/claude-x#mini-swe-agent"]
    assert model_status == {}


def test_print_pool_table_lists_identity_rows(tmp_path, capsys):
    store = _store_with_identity_pools(tmp_path)
    print_pool_table(store, ["flask"])
    out = capsys.readouterr().out
    assert "anthropic/claude-x [mini-swe-agent]" in out
    assert "effort=high" in out
    assert "provider=cloudflare" in out
