"""swe-duel-tournament-as (run_tournament_active_sampling) unit tests.

Offline coverage for the command's pure helpers: rankings.json loading,
participant config building, the challenge-slot reuse plan (admitted /
exhausted / to-generate), prior-match scoping, the active-sampling wiring
(budget + already-played exclusion), the --resume-state helpers (state
location/validation + rankings-export agreement), and the end-of-flow
contribution zip (states / claimed matches / the challenges+defenses
those matches' turns reference / the entrants' failed attempts / the
whole ./config directory, verified
round-trip through the swe-duel-tournament-update extraction). No Docker
or LLM is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.cli.run_tournament_active_sampling import (
    _build_model_configs,
    _load_prior_matches,
    _load_rankings,
    _prepare_contribution_zip,
    _resolve_newcomers,
    _slot_plan,
    resolve_rankings_selection,
)
from swe_duel.config import ModelConfig
from swe_duel.models import MatchOutcome
from swe_duel.scoring.active_sampling import select_intake_pairings

# ── fixtures ───────────────────────────────────────────────────

_TWO_ENTRIES = [
    {
        "rank": 1,
        "model": "openai/gpt-5.5",
        "harness": "codex",
        "points": 8.5,
        "reasoning_effort": "medium",
        "provider": "(OpenRouter auto-route)",
        "bradley_terry": 2.819,
        "elo": 1589.9,
    },
    {
        "rank": 2,
        "model": "anthropic/claude-opus-4.8",
        "harness": "mini-swe-agent",
        "points": 8.0,
        "reasoning_effort": "high",
        "provider": "(OpenRouter auto-route)",
        "bradley_terry": 2.461,
        "elo": 1582.1,
    },
]


@pytest.fixture
def rankings_file(tmp_path: Path) -> Path:
    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps(
            {
                "tournament_id": "tid",
                "format": "round_robin",
                "rankings": _TWO_ENTRIES,
            }
        )
    )
    return path


# ── _load_rankings ─────────────────────────────────────────────


def test_load_rankings_pins_recorded_defaults(rankings_file: Path) -> None:
    data = _load_rankings(rankings_file)
    # The entries' reasoning_effort / provider columns record the OpenRouter
    # defaults the original runs used; pinning them gives the participants
    # the exact identity the migrated bank/defense data is marked with.
    assert [e["cid"] for e in data.entries] == [
        "openai/gpt-5.5#codex#medium#",
        "anthropic/claude-opus-4.8#mini-swe-agent#high#",
    ]
    assert data.entries[0]["rank"] == 1
    assert data.entries[0]["points"] == 8.5
    assert data.entries[1]["bradley_terry"] == 2.461
    # A legacy export without repo/target fields tolerates their absence.
    assert data.repos == []
    assert data.targets_per_repo is None


def test_load_rankings_null_effort_keeps_two_part_form(tmp_path: Path) -> None:
    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps(
            {
                "rankings": [
                    {
                        "model": "moonshotai/kimi-k2.7-code",
                        "harness": "mini-swe-agent",
                        "reasoning_effort": None,
                        "provider": "(OpenRouter auto-route)",
                    }
                ]
            }
        )
    )
    data = _load_rankings(path)
    assert [e["cid"] for e in data.entries] == ["moonshotai/kimi-k2.7-code#mini-swe-agent"]


def test_load_rankings_claude_code_cannot_pin_effort(tmp_path: Path) -> None:
    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps(
            {
                "rankings": [
                    {
                        "model": "anthropic/claude-opus-4.8",
                        "harness": "claude-code",
                        "reasoning_effort": "high",
                        "provider": "(OpenRouter auto-route)",
                    }
                ]
            }
        )
    )
    data = _load_rankings(path)
    # claude-code has no reasoning-effort control: the informational column
    # is dropped and the identity stays the 2-part form (model default).
    assert [e["cid"] for e in data.entries] == ["anthropic/claude-opus-4.8#claude-code"]


def test_load_rankings_rejects_codex_effort_outside_enum(tmp_path: Path) -> None:
    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps(
            {
                "rankings": [
                    {
                        "model": "openai/gpt-5.5",
                        "harness": "codex",
                        "reasoning_effort": "max",
                    }
                ]
            }
        )
    )
    with pytest.raises(SystemExit, match="cannot express reasoning effort"):
        _load_rankings(path)


# ── resolve_rankings_selection ─────────────────────────────────


def test_resolve_selection_auto_route_provider_is_empty() -> None:
    assert resolve_rankings_selection(
        "mini-swe-agent", "high", "(OpenRouter auto-route)", "m#h"
    ) == ("high", "")
    assert resolve_rankings_selection(
        "mini-swe-agent", None, None, "m#h"
    ) == ("", "")


def test_resolve_selection_provider_slug_pins_or_fails_fast() -> None:
    # mini-swe-agent can pin providers.
    assert resolve_rankings_selection(
        "mini-swe-agent", "", "z-ai", "m#h"
    ) == ("", "z-ai")
    # codex / claude-code cannot.
    with pytest.raises(SystemExit, match="cannot pin provider"):
        resolve_rankings_selection("codex", "", "z-ai", "m#h")
    with pytest.raises(SystemExit, match="cannot pin provider"):
        resolve_rankings_selection("claude-code", "", "z-ai", "m#h")


def test_load_rankings_parses_repos_and_targets(tmp_path: Path) -> None:
    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps(
            {
                "rankings": _TWO_ENTRIES,
                "repos": ["jwt", "flask", "jwt", "flask"],
                "targets_per_repo": 1,
            }
        )
    )
    data = _load_rankings(path)
    assert data.targets_per_repo == 1
    # Order preserved, duplicates collapsed.
    assert data.repos == ["jwt", "flask"]


def test_load_rankings_rejects_bad_repos(tmp_path: Path) -> None:
    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps({"rankings": _TWO_ENTRIES, "repos": ["flask", 42]})
    )
    with pytest.raises(SystemExit, match="'repos' must be a list"):
        _load_rankings(path)


@pytest.mark.parametrize("bad", [0, -1, "1", 1.5, True])
def test_load_rankings_rejects_bad_targets(tmp_path: Path, bad: object) -> None:
    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps({"rankings": _TWO_ENTRIES, "targets_per_repo": bad})
    )
    with pytest.raises(SystemExit, match="targets_per_repo"):
        _load_rankings(path)


def test_load_rankings_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="rankings file not found"):
        _load_rankings(tmp_path / "nope.json")


def test_load_rankings_rejects_unknown_harness(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {"rankings": [{"model": "openai/gpt-5.5", "harness": "bogus-harness"}]}
        )
    )
    with pytest.raises(SystemExit, match="unknown harness"):
        _load_rankings(path)


def test_load_rankings_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "dup.json"
    path.write_text(json.dumps({"rankings": _TWO_ENTRIES + [_TWO_ENTRIES[0]]}))
    with pytest.raises(SystemExit, match="duplicate participant"):
        _load_rankings(path)


def test_load_rankings_rejects_empty_array(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"rankings": []}))
    with pytest.raises(SystemExit, match="no non-empty 'rankings' array"):
        _load_rankings(path)


# ── _build_model_configs ───────────────────────────────────────


def test_build_model_configs_reuses_curated_and_autoconstructs() -> None:
    curated = ModelConfig(
        model_id="z-ai/glm-5.2",
        max_tokens=1024,
        reasoning_efforts=["high", "low"],
        providers=["z-ai"],
    )
    cids = [
        "z-ai/glm-5.2#mini-swe-agent",
        "openai/gpt-5.5#codex",
        "anthropic/claude-opus-4.8#mini-swe-agent#high#",
    ]
    cfgs = _build_model_configs(
        {"glm-5.2": curated}, cids, max_tokens=32768
    )
    # Curated entry reused (2-part selection → base behaviour).
    assert cfgs["z-ai/glm-5.2#mini-swe-agent"].model_id == "z-ai/glm-5.2"
    assert cfgs["z-ai/glm-5.2#mini-swe-agent"].reasoning_efforts == ["high", "low"]
    # Rankings-only model auto-constructed: arena max_tokens, empty menus, and
    # the model-default effort + auto-route identity.
    auto = cfgs["openai/gpt-5.5#codex"]
    assert auto.model_id == "openai/gpt-5.5"
    assert auto.max_tokens == 32768
    assert auto.reasoning_efforts == []
    assert auto.providers == []
    assert auto.reasoning_effort == ""
    assert auto.provider == ""
    # The pinned selection from the composite id reaches the config of BOTH
    # the curated and the auto-constructed path (it is competitor identity —
    # it must flow into the harness and the persisted records).
    pinned_curated = cfgs["anthropic/claude-opus-4.8#mini-swe-agent#high#"]
    assert pinned_curated.reasoning_effort == "high"
    assert pinned_curated.provider == ""
    assert pinned_curated.model_id == "anthropic/claude-opus-4.8"


def test_rankings_pinned_selection_reaches_defense_config_lookup(
    tmp_path: Path,
) -> None:
    """Offline proof that defense runs use the effort/provider stored in
    rankings_<id>.json: the export's columns are pinned into the participant
    cid at load time, ``_build_model_configs`` binds them onto the cid-keyed
    ModelConfig, and MatchOrchestrator resolves exactly that config for the
    Blue defender (``_resolve_model_config``'s exact-cid path — the config
    ``run_defense_task`` hands to ``get_harness``, whose request body carries
    the selection)."""
    from unittest.mock import MagicMock

    from swe_duel.config import ArenaConfig
    from swe_duel.engine.match import MatchOrchestrator
    from swe_duel.logging.artifacts import ArtifactLogger

    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps(
            {
                "rankings": [
                    {
                        "rank": 1,
                        "model": "z-ai/glm-5.2",
                        "harness": "mini-swe-agent",
                        "reasoning_effort": "high",
                        "provider": "z-ai",
                    }
                ]
            }
        )
    )
    data = _load_rankings(path)
    cid = data.entries[0]["cid"]
    assert cid == "z-ai/glm-5.2#mini-swe-agent#high#z-ai"

    model_configs = _build_model_configs({}, [cid], max_tokens=4096)
    # The rankings-only model is auto-constructed AND pinned to the recorded
    # selection (it is competitor identity, not just a pool-key detail).
    assert model_configs[cid].reasoning_effort == "high"
    assert model_configs[cid].provider == "z-ai"

    # The orchestrator the command builds (cid-keyed configs) resolves the
    # same pinned config when constructing the Blue defender's harness.
    orchestrator = MatchOrchestrator(
        model_configs=model_configs,
        challenge_store=ChallengeStore(bank_dir=tmp_path / "bank"),
        workspace_manager=MagicMock(),
        config=ArenaConfig(),
        artifact_logger=ArtifactLogger(data_dir=tmp_path / "data"),
    )
    resolved = orchestrator._resolve_model_config(
        "z-ai/glm-5.2", "mini-swe-agent", "high", "z-ai"
    )
    assert resolved is model_configs[cid]
    assert resolved.reasoning_effort == "high"
    assert resolved.provider == "z-ai"


# ── _resolve_newcomers ─────────────────────────────────────────


def test_resolve_newcomers_accepts_newcomer_composites() -> None:
    field = ["openai/gpt-5.5#codex#medium#"]
    by_cid = {
        "openai/gpt-5.5#codex#medium#": {
            "model": "openai/gpt-5.5", "harness": "codex",
        },
    }
    # A valid composite id that is not an export participant is a new
    # entrant (unrated join), not an error.
    assert _resolve_newcomers(
        ["z-ai/glm-5.3#mini-swe-agent#max#z-ai"], field, by_cid
    ) == ["z-ai/glm-5.3#mini-swe-agent#max#z-ai"]
    # Newcomer ids go through the same capability rules: a claude-code
    # newcomer cannot pin an effort (dropped to the 2-part form).
    assert _resolve_newcomers(
        ["anthropic/claude-opus-4.8#claude-code#high#"], field, by_cid
    ) == ["anthropic/claude-opus-4.8#claude-code"]
    # Duplicates collapse.
    assert _resolve_newcomers(
        ["z-ai/glm-5.3#mini-swe-agent#max#z-ai"] * 2, field, by_cid
    ) == ["z-ai/glm-5.3#mini-swe-agent#max#z-ai"]
    # A codex newcomer with an inexpressible effort fails fast.
    with pytest.raises(SystemExit, match="cannot express reasoning effort"):
        _resolve_newcomers(["openai/o3#codex#ultrathink#"], field, by_cid)


def test_resolve_newcomers_rejects_field_members() -> None:
    field = [
        "openai/gpt-5.5#codex#medium#",
        "anthropic/claude-opus-4.8#claude-code",
    ]
    by_cid = {
        "openai/gpt-5.5#codex#medium#": {
            "model": "openai/gpt-5.5", "harness": "codex",
        },
        "anthropic/claude-opus-4.8#claude-code": {
            "model": "anthropic/claude-opus-4.8", "harness": "claude-code",
        },
    }
    # Active sampling always covers the whole field, so naming a rankings
    # participant (exact pinned id or bare model#harness form) is rejected
    # with the new semantics instead of (de)selecting it.
    with pytest.raises(SystemExit, match="already in the rankings field"):
        _resolve_newcomers(["openai/gpt-5.5#codex#medium#"], field, by_cid)
    with pytest.raises(SystemExit, match="already in the rankings field"):
        _resolve_newcomers(["anthropic/claude-opus-4.8#claude-code"], field, by_cid)
    with pytest.raises(SystemExit, match="already in the rankings field"):
        _resolve_newcomers(["openai/gpt-5.5#codex"], field, by_cid)
    # A composite that parses down to an existing participant (claude-code
    # cannot pin effort/provider, so the pins collapse away) is a field
    # member too.
    with pytest.raises(SystemExit, match="resolves to"):
        _resolve_newcomers(["anthropic/claude-opus-4.8#claude-code#high#"], field, by_cid)


def test_resolve_newcomers_rejects_unknown_ids() -> None:
    field = ["openai/gpt-5.5#codex#medium#"]
    by_cid = {
        "openai/gpt-5.5#codex#medium#": {
            "model": "openai/gpt-5.5", "harness": "codex",
        },
    }
    # An unknown harness is not a parseable entrant.
    with pytest.raises(SystemExit, match="new-entrant"):
        _resolve_newcomers(["openai/gpt-5.5#not-a-harness"], field, by_cid)
    # A bare model id (no harness) is not a newcomer either.
    with pytest.raises(SystemExit, match="new-entrant"):
        _resolve_newcomers(["openai/gpt-5.5"], field, by_cid)


# ── _resolve_rankings_path / _enforce_exported_arena ───────────


def test_resolve_rankings_path_requires_a_choice(tmp_path: Path) -> None:
    from swe_duel.cli import run_tournament_active_sampling as m

    args = argparse.Namespace(tournament=None, rankings=None)
    with pytest.raises(SystemExit, match="choose which tournament to enter"):
        m._resolve_rankings_path(args)


def test_resolve_rankings_path_tournament_resolves_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swe_duel.cli import run_tournament_active_sampling as m

    monkeypatch.chdir(tmp_path)
    (tmp_path / "rankings").mkdir()
    (tmp_path / "rankings" / "rankings_tid-1.json").write_text("{}")
    args = argparse.Namespace(tournament="tid-1", rankings=None)
    assert m._resolve_rankings_path(args) == Path(
        "./rankings/rankings_tid-1.json"
    )
    # Unknown tournament id lists what exists.
    args = argparse.Namespace(tournament="nope", rankings=None)
    with pytest.raises(SystemExit, match="tid-1"):
        m._resolve_rankings_path(args)
    # The two selectors are mutually exclusive.
    args = argparse.Namespace(tournament="tid-1", rankings="x.json")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        m._resolve_rankings_path(args)


def test_enforce_exported_arena_rejects_differing_values() -> None:
    from swe_duel.cli import run_tournament_active_sampling as m

    field_data = m.RankingsField(
        entries=[],
        repos=["flask", "jinja"],
        targets_per_repo=1,
    )
    # Same set (any order) and same target are accepted.
    m._enforce_exported_arena(
        argparse.Namespace(repos=["jinja", "flask"], turns_per_player=1),
        field_data,
        Path("r.json"),
    )
    # A differing repo set biases the arena — rejected.
    with pytest.raises(SystemExit, match="repos"):
        m._enforce_exported_arena(
            argparse.Namespace(repos=["flask"], turns_per_player=None),
            field_data,
            Path("r.json"),
        )
    # A differing challenge target is rejected too.
    with pytest.raises(SystemExit, match="targets_per_repo=1"):
        m._enforce_exported_arena(
            argparse.Namespace(repos=None, turns_per_player=3),
            field_data,
            Path("r.json"),
        )
    # A legacy export without the fields keeps CLI-override behaviour.
    m._enforce_exported_arena(
        argparse.Namespace(repos=["flask"], turns_per_player=7),
        m.RankingsField(entries=[]),
        Path("r.json"),
    )


# ── _slot_plan ─────────────────────────────────────────────────


def _success_record(store: ChallengeStore, model: str, repo: str, slot: int) -> None:
    from swe_duel.models import (
        AgentTrajectory,
        ChallengeRecord,
        RedChallenge,
        RedValidationResult,
    )

    traj = AgentTrajectory(
        steps=[], total_steps=0, total_input_tokens=0, total_output_tokens=0,
        total_cost_usd=0.0, model_id=model, duration_seconds=0.0,
    )
    record = ChallengeRecord(
        challenge_id=str(uuid.uuid4()),
        red_model_id=model,
        repo_name=repo,
        repo_commit_sha="abc",
        target_files=["src/foo.py"],
        challenge=RedChallenge(
            target_files=["src/foo.py"], exploration_summary="", feature_spec="",
            feature_rationale="", pr_diff="", modified_file_contents={},
            original_file_contents={}, feature_test_code="",
            bug_type=None, bug_description=None, bug_location=None,
            bug_test_code=None, agent_trajectory=traj,
        ),
        validation=RedValidationResult(passed=True, gate_results=[], attempt_number=1),
        generated_at=datetime(2026, 4, 19, 12, 0, 0),
        generation_cost_usd=0.0,
        generation_retries=0,
        red_harness_id="mini-swe-agent",
        slot=slot,
    )
    store.store(record)


def _failed_record(
    store: ChallengeStore, model: str, repo: str, slot: int, attempt: int
) -> None:
    from swe_duel.models import FailedChallengeRecord

    frec = FailedChallengeRecord(
        challenge_id=str(uuid.uuid4()),
        red_model_id=model,
        repo_name=repo,
        repo_commit_sha="abc",
        kind="validation",
        error_message="gate_bug_tests=failed",
        attempt_number=attempt,
        target_files=[],
        challenge=None,
        validation=None,
        feature_trajectory=None,
        bug_trajectory=None,
        self_review_trajectory=None,
        generated_at=datetime(2026, 4, 19, 12, 0, 0),
        generation_cost_usd=0.0,
        elapsed_seconds=1.0,
        red_harness_id="mini-swe-agent",
        slot=slot,
    )
    store.store_failed(frec)


def test_slot_plan_splits_admitted_exhausted_and_new(tmp_path: Path) -> None:
    store = ChallengeStore(bank_dir=tmp_path / "bank")
    cid = "openai/gpt-5.5#mini-swe-agent"
    model = "openai/gpt-5.5"
    # Slot 1: admitted challenge. Slot 2: exhausted 5-attempt failure chain.
    _success_record(store, model, "flask", 1)
    for attempt in range(1, 6):
        _failed_record(store, model, "flask", 2, attempt)
    # Slot 3: never attempted.

    plan = _slot_plan(store, cid, ["flask", "jwt"], turns_per_player=3)
    flask = plan["flask"]
    assert flask["admitted"] == [1]
    assert flask["exhausted"] == [2]
    assert flask["to_generate"] == [3]
    # An untouched pool has every slot to generate.
    assert plan["jwt"]["to_generate"] == [1, 2, 3]
    assert plan["jwt"]["admitted"] == []
    assert plan["jwt"]["exhausted"] == []


def test_slot_plan_zero_turns_means_nothing_to_generate(tmp_path: Path) -> None:
    store = ChallengeStore(bank_dir=tmp_path / "bank")
    plan = _slot_plan(store, "openai/gpt-5.5#codex", ["flask"], turns_per_player=0)
    assert plan["flask"] == {"admitted": [], "exhausted": [], "to_generate": []}


# ── _load_prior_matches ────────────────────────────────────────


def _write_match(
    matches_dir: Path, a: str, b: str, outcome: str, mid: str | None = None
) -> None:
    payload = {
        "match_id": mid or str(uuid.uuid4()),
        "model_a_id": a,
        "model_b_id": b,
        "repo_name": "flask",
        "outcome": outcome,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (matches_dir / f"{payload['match_id']}.json").write_text(json.dumps(payload))


def test_load_prior_matches_scopes_to_field(tmp_path: Path) -> None:
    matches_dir = tmp_path / "matches"
    matches_dir.mkdir()
    field = {"a#mini-swe-agent", "b#mini-swe-agent", "c#mini-swe-agent"}
    _write_match(matches_dir, "a#mini-swe-agent", "b#mini-swe-agent", "model_a_wins")
    _write_match(matches_dir, "b#mini-swe-agent", "c#mini-swe-agent", "draw")
    # Foreign participant + unparseable file are ignored.
    _write_match(matches_dir, "a#mini-swe-agent", "zzz#mini-swe-agent", "draw")
    (matches_dir / "broken.json").write_text("{not json")

    matches, played, secondary = _load_prior_matches(matches_dir, field)
    assert len(matches) == 2
    assert len(played) == 2
    assert {m.outcome for m in matches} <= {MatchOutcome.MODEL_A_WINS, MatchOutcome.DRAW}
    # TrueSkill-σ tie-break map covers exactly the field members that played.
    assert set(secondary) == {"a#mini-swe-agent", "b#mini-swe-agent", "c#mini-swe-agent"}
    assert all(v > 0 for v in secondary.values())


def test_load_prior_matches_empty_dir(tmp_path: Path) -> None:
    matches, played, secondary = _load_prior_matches(tmp_path / "missing", set())
    assert matches == []
    assert played == set()
    assert secondary == {}


def test_load_prior_matches_merges_export_head_to_head(tmp_path: Path) -> None:
    # Local match file + export-supplied match that only exists in the
    # rankings file → both count, deduped by match_id.
    matches_dir = tmp_path / "matches"
    matches_dir.mkdir()
    field = {"a#mini-swe-agent", "b#mini-swe-agent", "c#mini-swe-agent"}
    _write_match(matches_dir, "a#mini-swe-agent", "b#mini-swe-agent", "model_a_wins")
    extra = [
        {
            "match_id": "m-export-1",
            "model_a": "b#mini-swe-agent",
            "model_b": "c#mini-swe-agent",
            "outcome": "model_b_wins",
            "timestamp": "2026-10-06T00:00:00+00:00",
            "repo_name": "flask",
        },
        # Same match_id as the local a-b file → local wins, no double count.
        {
            "match_id": None,
            "model_a": "a#mini-swe-agent",
            "model_b": "b#mini-swe-agent",
            "outcome": "draw",
            "timestamp": "2026-10-06T00:00:00+00:00",
        },
        # Foreign pair in the export list is ignored like foreign files are.
        {
            "match_id": "m-export-2",
            "model_a": "a#mini-swe-agent",
            "model_b": "zzz#mini-swe-agent",
            "outcome": "draw",
        },
    ]
    matches, played, secondary = _load_prior_matches(
        matches_dir, field, extra_payloads=extra
    )
    assert len(matches) == 2
    assert {m.match_id for m in matches} == {"m-export-1", matches[0].match_id}
    assert len(played) == 2
    assert set(secondary) == {"a#mini-swe-agent", "b#mini-swe-agent", "c#mini-swe-agent"}


def test_load_rankings_parses_export_matches(tmp_path: Path) -> None:
    path = tmp_path / "rankings_tid.json"
    path.write_text(
        json.dumps(
            {
                "rankings": [
                    {"model": "a", "harness": "mini-swe-agent"},
                    {"model": "b", "harness": "mini-swe-agent"},
                ],
                "matches": [
                    {
                        "match_id": "m1",
                        "model_a": "a#mini-swe-agent",
                        "model_b": "b#mini-swe-agent",
                        "outcome": "draw",
                    }
                ],
            }
        )
    )
    data = _load_rankings(path)
    assert data.matches == [
        {
            "match_id": "m1",
            "model_a": "a#mini-swe-agent",
            "model_b": "b#mini-swe-agent",
            "outcome": "draw",
        }
    ]
    # Malformed match entries fail fast.
    path.write_text(
        json.dumps(
            {
                "rankings": [{"model": "a", "harness": "mini-swe-agent"}],
                "matches": [{"match_id": "m1", "model_a": "a#mini-swe-agent"}],
            }
        )
    )
    with pytest.raises(SystemExit, match="model_b"):
        _load_rankings(path)


# ── active-sampling wiring ─────────────────────────────────────


def test_sampling_budget_and_already_played_exclusion() -> None:
    field = ["a#mini-swe-agent", "b#mini-swe-agent", "c#mini-swe-agent",
             "d#mini-swe-agent"]
    targets = ["a#mini-swe-agent"]
    # Fresh data dir: nothing played, budget 2 → two distinct opponents.
    pairings = select_intake_pairings(
        newcomers=targets,
        all_players=field,
        already_played=set(),
        remaining_budget={targets[0]: 2},
        matches=[],
        secondary_score={},
    )
    assert len(pairings) == 2
    assert all(p.model_a == targets[0] or p.model_b == targets[0] for p in pairings)
    opponents = {
        p.model_b if p.model_a == targets[0] else p.model_a for p in pairings
    }
    assert len(opponents) == 2

    # With a-b already played, the sampler moves to the next opponents.
    from swe_duel.scoring.active_sampling import pair_key

    pairings2 = select_intake_pairings(
        newcomers=targets,
        all_players=field,
        already_played={pair_key(targets[0], "b#mini-swe-agent")},
        remaining_budget={targets[0]: 2},
        matches=[],
        secondary_score={},
    )
    opponents2 = {
        p.model_b if p.model_a == targets[0] else p.model_a for p in pairings2
    }
    assert "b#mini-swe-agent" not in opponents2
    assert len(pairings2) == 2


# ── _prepare_contribution_zip ───────────────────────────────────

_ZIP_TID = "tid-2222"
_ZIP_A = "openai/gpt-5.5#codex#medium#"
_ZIP_B = "anthropic/claude-opus-4.8#mini-swe-agent#high#"
_ZIP_N = "z-ai/glm-5.3-flash#mini-swe-agent#high#z-ai"
_ZIP_FOREIGN = "moonshotai/kimi-k2.7-code#mini-swe-agent"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _zip_match_payload(
    mid: str,
    a: str,
    b: str,
    repo: str = "flask",
    turns: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "match_id": mid,
        "model_a_id": a,
        "model_b_id": b,
        "repo_name": repo,
        "outcome": "draw",
        "timestamp": "2026-10-06T00:00:00+00:00",
        "turns": turns or [],
    }


def _zip_turn(challenge: str, defense: str) -> dict[str, Any]:
    """A turn record as serialized into a match file — the packaging
    scope derives the matchup-relevant challenge/defense ids from these."""
    return {"turn_index": 0, "challenge_id": challenge, "defense_id": defense}


def _zip_auto_win_turn() -> dict[str, Any]:
    """A missing-Red auto-win turn as serialized into a match file: an
    empty ``challenge_id`` plus a SYNTHETIC defense_id and the forfeit-
    perfect score (see MatchOrchestrator._auto_win_turn). No defense ever
    ran, so no record/log/workspace ever existed — the packaging scope
    must neither package nor warn about the synthetic id."""
    return {
        "turn_index": 1,
        "challenge_id": "",
        "defense_id": "d-auto-win",
        "score": {
            "s_regression": 1.0,
            "s_feature": 1.0,
            "s_bugfix": 1.0,
            "blue_composite": 1.0,
            "red_composite": 0.0,
            "test_details": {"auto_win": True, "reason": "red_missing_challenge"},
        },
    }


def _zip_state_payload(name: str, entered: str | None, *, source: str = "") -> dict[str, Any]:
    return {
        "tournament_id": name,
        "format": "active_sampling",
        "entered_tournament": entered,
        "rankings_source": source,
        "repo_names": ["flask"],
        "targets_per_repo": 1,
        "field": [_ZIP_A, _ZIP_B, _ZIP_N],
        "targets": [_ZIP_N],
        "newcomers": [_ZIP_N],
        "budget": 1,
        "pairings": [{"model_a": _ZIP_N, "model_b": _ZIP_A}],
        "results": [
            {
                "match_id": f"m-{name}",
                "model_a": _ZIP_N,
                "model_b": _ZIP_A,
                "outcome": "model_a_wins",
            }
        ],
        "status": "complete",
        "timestamp": "2026-10-06T00:00:00+00:00",
    }


@pytest.fixture
def contributor_data(tmp_path: Path) -> Path:
    """A two-session contributor's ``./data`` plus unrelated noise.

    Session ``this`` and session ``earlier`` both entered ``tid-2222``; a
    third state entered another tournament. The bank/defenses carry the
    records the claimed matches' **turns reference** (with html twins)
    plus in-field-but-unreferenced and foreign records that must stay out
    of the zip — the incumbents' pre-existing data is the organizer's
    already; ``workspaces``/``logs`` are never collected.
    """
    data = tmp_path / "data"
    tdir = data / "tournaments"
    _write_json(
        tdir / "active_sampling_state_this.json",
        _zip_state_payload("this", _ZIP_TID),
    )
    # An earlier session of the same newcomer against a different incumbent.
    earlier = _zip_state_payload("earlier", _ZIP_TID)
    earlier["pairings"] = [{"model_a": _ZIP_N, "model_b": _ZIP_B}]
    earlier["results"] = [
        {
            "match_id": "m-earlier",
            "model_a": _ZIP_N,
            "model_b": _ZIP_B,
            "outcome": "draw",
        }
    ]
    _write_json(tdir / "active_sampling_state_earlier.json", earlier)
    # A run that entered a DIFFERENT tournament: out of scope (its pairing
    # and repo match no selected state's schedule).
    other = _zip_state_payload("other", "tid-9999")
    other["field"] = [_ZIP_A, _ZIP_FOREIGN, _ZIP_N]
    other["repo_names"] = ["jwt"]
    other["pairings"] = [{"model_a": _ZIP_N, "model_b": _ZIP_FOREIGN}]
    other["results"] = [
        {
            "match_id": "m-other",
            "model_a": _ZIP_N,
            "model_b": _ZIP_FOREIGN,
            "outcome": "draw",
        }
    ]
    _write_json(tdir / "active_sampling_state_other.json", other)
    # Matches: the two same-tournament claims are in; the other-tournament
    # match and an incumbent-vs-incumbent match among no state's pairings
    # stay out. Each claimed match's turn references the records the zip
    # must scope to (c-n + the defending side's defense).
    mdir = data / "matches"
    _write_json(
        mdir / "m-this.json",
        _zip_match_payload(
            "m-this",
            _ZIP_N,
            _ZIP_A,
            turns=[_zip_turn("c-n", "d-a"), _zip_auto_win_turn()],
        ),
    )
    _write_json(
        mdir / "m-earlier.json",
        _zip_match_payload(
            "m-earlier", _ZIP_N, _ZIP_B, turns=[_zip_turn("c-n", "d-b")]
        ),
    )
    _write_json(
        mdir / "m-other.json",
        _zip_match_payload("m-other", _ZIP_N, _ZIP_FOREIGN, repo="jwt"),
    )
    _write_json(mdir / "m-incumbents.json", _zip_match_payload("m-incumbents", _ZIP_A, _ZIP_B))
    # Challenges: turn-referenced (packages), in-field-but-unreferenced
    # (stays out — the incumbents' bank is the organizer's data already),
    # foreign identity (stays out).
    ch = _write_json(
        data / "challenge_bank" / "challenges" / "c-n.json",
        {
            "challenge_id": "c-n",
            "red_model_id": "z-ai/glm-5.3-flash",
            "red_harness_id": "mini-swe-agent",
            "red_reasoning_effort": "high",
            "red_provider": "z-ai",
            "repo_name": "flask",
        },
    )
    ch.with_suffix(".html").write_text("<html>c-n</html>")
    _write_json(
        data / "challenge_bank" / "challenges" / "c-unplayed.json",
        {
            "challenge_id": "c-unplayed",
            "red_model_id": "z-ai/glm-5.3-flash",
            "red_harness_id": "mini-swe-agent",
            "red_reasoning_effort": "high",
            "red_provider": "z-ai",
            "repo_name": "flask",
        },
    )
    _write_json(
        data / "challenge_bank" / "challenges" / "c-foreign.json",
        {
            "challenge_id": "c-foreign",
            "red_model_id": "moonshotai/kimi-k2.7-code",
            "red_harness_id": "mini-swe-agent",
            "red_reasoning_effort": "",
            "red_provider": "",
            "repo_name": "flask",
        },
    )
    # Failed attempts: the entrant's own chain packages; an incumbent's
    # chain is the organizer's data already and stays out.
    _write_json(
        data / "challenge_bank" / "failed_challenges" / "f-n.json",
        {
            "challenge_id": "f-n",
            "red_model_id": "z-ai/glm-5.3-flash",
            "red_harness_id": "mini-swe-agent",
            "red_reasoning_effort": "high",
            "red_provider": "z-ai",
            "repo_name": "flask",
        },
    )
    _write_json(
        data / "challenge_bank" / "failed_challenges" / "f-a.json",
        {
            "challenge_id": "f-a",
            "red_model_id": "openai/gpt-5.5",
            "red_harness_id": "codex",
            "red_reasoning_effort": "medium",
            "red_provider": "",
            "repo_name": "flask",
        },
    )
    # Defenses: turn-referenced (package, with twins), in-field-but-
    # unreferenced (stays out), foreign identity (stays out).
    d = _write_json(
        data / "defenses" / "d-a.json",
        {
            "defense_id": "d-a",
            "challenge_id": "c-n",
            "blue_model_id": "openai/gpt-5.5",
            "blue_harness_id": "codex",
            "blue_reasoning_effort": "medium",
            "blue_provider": "",
        },
    )
    d.with_suffix(".html").write_text("<html>d-a</html>")
    d = _write_json(
        data / "defenses" / "d-b.json",
        {
            "defense_id": "d-b",
            "challenge_id": "c-n",
            "blue_model_id": "anthropic/claude-opus-4.8",
            "blue_harness_id": "mini-swe-agent",
            "blue_reasoning_effort": "high",
            "blue_provider": "",
        },
    )
    d.with_suffix(".html").write_text("<html>d-b</html>")
    _write_json(
        data / "defenses" / "d-unplayed.json",
        {
            "defense_id": "d-unplayed",
            "challenge_id": "c-unplayed",
            "blue_model_id": "anthropic/claude-opus-4.8",
            "blue_harness_id": "mini-swe-agent",
            "blue_reasoning_effort": "high",
            "blue_provider": "",
        },
    )
    _write_json(
        data / "defenses" / "d-foreign.json",
        {
            "defense_id": "d-foreign",
            "challenge_id": "c-foreign",
            "blue_model_id": "moonshotai/kimi-k2.7-code",
            "blue_harness_id": "mini-swe-agent",
            "blue_reasoning_effort": "",
            "blue_provider": "",
        },
    )
    _write_json(
        data / "challenge_bank" / "index.json",
        {"pools": {}, "failed_pools": {}, "entries": {}},
    )
    # Never part of a curated contribution zip.
    _write_json(data / "workspaces" / "ws.json", {})
    _write_json(data / "logs" / "run.log", {})
    return data


@pytest.fixture
def contributor_config(tmp_path: Path) -> Path:
    """A contributor's ``./config`` — the editable deployment configuration
    (arena/models/repos YAML) that rides along under the config/ archive
    root of every contribution zip."""
    config = tmp_path / "config"
    (config / "repos").mkdir(parents=True)
    (config / "arena.yaml").write_text("match:\n  turns_per_player: 1\n")
    (config / "models.yaml").write_text("models: []\n")
    (config / "repos" / "flask.yaml").write_text("name: flask\n")
    return config


def test_prepare_contribution_zip_scopes_to_entered_tournament(
    contributor_data: Path,
    contributor_config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path = contributor_data / "tournaments" / "active_sampling_state_this.json"
    state_payload = json.loads(state_path.read_text())
    out_dir = tmp_path / "submissions"
    target = tmp_path / "fixed.zip"

    got = _prepare_contribution_zip(
        contributor_data,
        state_path,
        state_payload,
        ["m-this"],
        out_dir,
        zip_path=target,
        config_dir=contributor_config,
    )
    assert got == target
    with zipfile.ZipFile(target) as zf:
        names = set(zf.namelist())
    # Both same-tournament states (this run's + the earlier session's), the
    # matches they claim, the challenge/defense records those matches'
    # turns reference (+ html twins), the entrant's failed chain, and the
    # bank index — nothing unreferenced, nothing foreign, no
    # workspaces/logs — plus the whole ./config directory under a
    # config/ archive root.
    assert names == {
        "data/tournaments/active_sampling_state_this.json",
        "data/tournaments/active_sampling_state_earlier.json",
        "data/matches/m-this.json",
        "data/matches/m-earlier.json",
        "data/challenge_bank/challenges/c-n.json",
        "data/challenge_bank/challenges/c-n.html",
        "data/challenge_bank/failed_challenges/f-n.json",
        "data/challenge_bank/index.json",
        "data/defenses/d-a.json",
        "data/defenses/d-a.html",
        "data/defenses/d-b.json",
        "data/defenses/d-b.html",
        "config/arena.yaml",
        "config/models.yaml",
        "config/repos/flask.yaml",
    }
    # The narrowing: in-field records the claimed matches' turns never
    # reference stay out — the incumbents' pre-existing bank/defenses are
    # the organizer's data already and only ballooned earlier zips.
    for absent in (
        "data/challenge_bank/challenges/c-unplayed.json",
        "data/challenge_bank/challenges/c-foreign.json",
        "data/challenge_bank/failed_challenges/f-a.json",
        "data/defenses/d-unplayed.json",
        "data/defenses/d-foreign.json",
        "data/matches/m-other.json",
        "data/matches/m-incumbents.json",
        # The auto-win turn's synthetic defense id references no record by
        # design (no defense ran) — it must never package nor warn.
        "data/defenses/d-auto-win.json",
    ):
        assert absent not in names
    captured = capsys.readouterr()
    assert "turn-referenced defense" not in captured.err
    assert "turn-referenced challenge" not in captured.err
    out = captured.out
    assert "SHA-256 " in out
    assert "swe-duel-tournament-update" in out
    assert "3 config file(s)" in out
    # Without an explicit zip_path the name encodes the entered tournament.
    auto = _prepare_contribution_zip(
        contributor_data,
        state_path,
        state_payload,
        ["m-this"],
        out_dir,
        config_dir=contributor_config,
    )
    assert auto.parent == out_dir
    assert re.fullmatch(rf"as_submission_{_ZIP_TID}_\d{{8}}-\d{{6}}\.zip", auto.name)


def test_verify_zip_members_rejects_irrelevant_records(
    contributor_data: Path,
) -> None:
    """The packaging verifier re-derives every member's relevance from
    disk: only selected states, their claimed matches, the records those
    matches' turns reference, and the new entrants' failed attempts ever
    package — workspaces/, unclaimed matches, unreferenced in-field
    records, foreign identities, orphan twins fail fast."""
    from swe_duel.cli import run_tournament_active_sampling as m

    data = contributor_data
    state = data / "tournaments" / "active_sampling_state_this.json"
    identities = {m._Identity.from_cid(c) for c in (_ZIP_A, _ZIP_B, _ZIP_N)}
    newcomers = {m._Identity.from_cid(_ZIP_N)}

    def _check(members: list[Path], **kw: Any) -> None:
        base: dict[str, Any] = dict(
            state_paths=[state],
            match_ids={"m-this", "m-earlier"},
            identities=identities,
            turn_challenge_ids={"c-n"},
            turn_defense_ids={"d-a", "d-b"},
            newcomer_identities=newcomers,
        )
        base.update(kw)
        m._verify_contribution_zip_members(data, members, **base)

    # The legit matchup-relevant member set verifies cleanly.
    _check(
        [
            state,
            data / "matches" / "m-this.json",
            data / "challenge_bank" / "challenges" / "c-n.json",
            data / "challenge_bank" / "failed_challenges" / "f-n.json",
            data / "defenses" / "d-a.json",
            data / "defenses" / "d-b.json",
            data / "challenge_bank" / "index.json",
        ]
    )
    # workspaces/ (and any non-record subtree) never packages.
    with pytest.raises(SystemExit, match="workspaces"):
        _check([data / "workspaces" / "ws.json"])
    # A match outside the states' claims is irrelevant.
    with pytest.raises(SystemExit, match="not one of the matches"):
        _check([data / "matches" / "m-incumbents.json"], match_ids={"m-this"})
    # A claimed match id whose file names a non-field participant is
    # irrelevant at the record level.
    _write_json(
        data / "matches" / "m-weird.json",
        _zip_match_payload("m-weird", _ZIP_N, _ZIP_FOREIGN),
    )
    with pytest.raises(SystemExit, match="match endpoint"):
        _check([data / "matches" / "m-weird.json"], match_ids={"m-weird"})
    # In-field identity is not enough: a challenge/defense the claimed
    # matches' turns never reference stays out.
    with pytest.raises(SystemExit, match="turns reference"):
        _check([data / "challenge_bank" / "challenges" / "c-unplayed.json"])
    with pytest.raises(SystemExit, match="turns reference"):
        _check([data / "defenses" / "d-unplayed.json"])
    # Foreign-identity records are never referenced either.
    with pytest.raises(SystemExit, match="turns reference"):
        _check([data / "challenge_bank" / "challenges" / "c-foreign.json"])
    with pytest.raises(SystemExit, match="turns reference"):
        _check([data / "defenses" / "d-foreign.json"])
    # An incumbent's failure chain is the organizer's data already.
    with pytest.raises(SystemExit, match="new entrants"):
        _check([data / "challenge_bank" / "failed_challenges" / "f-a.json"])
    # A state file outside the selected same-tournament set is irrelevant.
    with pytest.raises(SystemExit, match="not one of the selected"):
        _check(
            [data / "tournaments" / "active_sampling_state_other.json"],
            state_paths=[],
        )
    # An html twin without its .json record never packages either.
    (data / "defenses" / "orphan.html").write_text("<html>orphan</html>")
    with pytest.raises(SystemExit, match="html twin"):
        _check([data / "defenses" / "orphan.html"])


def test_verify_written_zip_matches_expected_members(tmp_path: Path) -> None:
    """The written archive is reopened and must contain exactly the verified
    member set — anything else (a workspaces/ leak) deletes the zip and
    fails fast."""
    from swe_duel.cli import run_tournament_active_sampling as m

    target = tmp_path / "bad.zip"
    with zipfile.ZipFile(target, "w") as zf:
        zf.writestr("data/matches/m.json", "{}")
        zf.writestr("data/workspaces/leak.bin", b"x" * 128)
    with pytest.raises(SystemExit, match="workspaces"):
        m._verify_written_zip(target, {"data/matches/m.json"})
    assert not target.exists()

    with zipfile.ZipFile(target, "w") as zf:
        zf.writestr("data/matches/m.json", "{}")
    m._verify_written_zip(target, {"data/matches/m.json"})
    assert target.exists()


def test_config_zip_members_walk_and_missing_dir(
    contributor_config: Path, tmp_path: Path
) -> None:
    """_config_zip_members collects every file under ./config (nested
    subtrees included) sorted by archive name; a missing directory fails
    fast instead of silently packaging nothing."""
    from swe_duel.cli import run_tournament_active_sampling as m

    got = m._config_zip_members(contributor_config)
    assert [p.relative_to(contributor_config).as_posix() for p in got] == [
        "arena.yaml",
        "models.yaml",
        "repos/flask.yaml",
    ]
    with pytest.raises(SystemExit, match="not a directory"):
        m._config_zip_members(tmp_path / "missing")


def test_verify_config_zip_members_requires_exact_mirror(
    contributor_config: Path, tmp_path: Path
) -> None:
    """The config verifier re-derives the expected set from disk: a path
    outside ./config, or a file added/deleted under it between the
    collection walk and the verification, fails fast."""
    from swe_duel.cli import run_tournament_active_sampling as m

    members = m._config_zip_members(contributor_config)
    m._verify_config_zip_members(contributor_config, members)

    outside = tmp_path / "secret.yaml"
    outside.write_text("nope\n")
    with pytest.raises(SystemExit, match="unexpected config member"):
        m._verify_config_zip_members(contributor_config, [*members, outside])
    # A file that appears under ./config after the walk is just as wrong.
    late = contributor_config / "late.yaml"
    late.write_text("late\n")
    with pytest.raises(SystemExit, match="missing config member"):
        m._verify_config_zip_members(contributor_config, members)
    # A file that disappears between walk and verify also fails fast.
    late.unlink()
    members.append(contributor_config / "late.yaml")
    with pytest.raises(SystemExit, match="unexpected config member"):
        m._verify_config_zip_members(contributor_config, members)


def test_prepared_zip_packages_config_verbatim(
    contributor_data: Path,
    contributor_config: Path,
    tmp_path: Path,
) -> None:
    """Every ./config file ships under a config/ archive root byte-for-byte
    next to the data/ record slice — the deployment configuration that
    shaped the packaged records."""
    state_path = contributor_data / "tournaments" / "active_sampling_state_this.json"
    target = tmp_path / "sub.zip"
    _prepare_contribution_zip(
        contributor_data,
        state_path,
        json.loads(state_path.read_text()),
        ["m-this"],
        tmp_path,
        zip_path=target,
        config_dir=contributor_config,
    )
    with zipfile.ZipFile(target) as zf:
        names = zf.namelist()
        assert "config/arena.yaml" in names
        assert "config/models.yaml" in names
        assert "config/repos/flask.yaml" in names
        assert (
            zf.read("config/arena.yaml")
            == (contributor_config / "arena.yaml").read_bytes()
        )
        # The data slice packages unchanged next to the config root.
        assert "data/tournaments/active_sampling_state_this.json" in names
        assert "data/challenge_bank/index.json" in names


def test_prepared_zip_excludes_workspaces(
    contributor_data: Path,
    contributor_config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """workspaces/ adds nothing but bulk: bury an incompressible leftover
    blob under data/workspaces/ and the prepared zip stays a records-only
    artifact (./config still rides along)."""
    blob = contributor_data / "workspaces" / "flask" / "ws-uuid" / "repo" / "big.bin"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(os.urandom(4 * 1024 * 1024))
    state_path = contributor_data / "tournaments" / "active_sampling_state_this.json"

    target = tmp_path / "sub.zip"
    _prepare_contribution_zip(
        contributor_data,
        state_path,
        json.loads(state_path.read_text()),
        ["m-this"],
        tmp_path,
        zip_path=target,
        config_dir=contributor_config,
    )
    with zipfile.ZipFile(target) as zf:
        names = zf.namelist()
    assert not any("workspaces" in n for n in names)
    assert not any("/logs/" in n for n in names)
    # The 4 MiB blob is absent, so the zip is only the small .json records.
    assert target.stat().st_size < 512 * 1024
    out = capsys.readouterr().out
    assert "workspaces/ and logs/ are excluded" in out
    assert "verified" in out


def test_prepare_contribution_zip_unlabeled_tournament_falls_back(
    tmp_path: Path,
) -> None:
    """Without a resolvable entered tournament (an explicit --rankings path
    that is not a ``rankings_<id>.json``), the scope is this run's state and
    the caller-supplied match ids only — other unlabeled states never get
    swept in."""
    data = tmp_path / "data"
    tdir = data / "tournaments"
    this = _write_json(
        tdir / "active_sampling_state_this.json",
        _zip_state_payload("this", None, source="./my_export.json"),
    )
    _write_json(
        tdir / "active_sampling_state_other.json",
        _zip_state_payload("other", None, source="./my_export.json"),
    )
    _write_json(
        data / "matches" / "m-this.json",
        _zip_match_payload(
            "m-this", _ZIP_N, _ZIP_A, turns=[_zip_turn("c-n", "d-x")]
        ),
    )
    _write_json(data / "matches" / "m-other.json", _zip_match_payload("m-other", _ZIP_N, _ZIP_B))
    _write_json(
        data / "challenge_bank" / "challenges" / "c-n.json",
        {
            "challenge_id": "c-n",
            "red_model_id": "z-ai/glm-5.3-flash",
            "red_harness_id": "mini-swe-agent",
            "red_reasoning_effort": "high",
            "red_provider": "z-ai",
            "repo_name": "flask",
        },
    )
    _write_json(
        data / "defenses" / "d-x.json",
        {
            "defense_id": "d-x",
            "challenge_id": "c-n",
            "blue_model_id": "openai/gpt-5.5",
            "blue_harness_id": "codex",
            "blue_reasoning_effort": "medium",
            "blue_provider": "",
        },
    )
    (tmp_path / "config").mkdir()

    target = tmp_path / "fixed.zip"
    _prepare_contribution_zip(
        data,
        this,
        json.loads(this.read_text()),
        # A match id with no file on disk is skipped silently.
        ["m-this", "m-missing"],
        tmp_path / "submissions",
        zip_path=target,
        config_dir=tmp_path / "config",
    )
    with zipfile.ZipFile(target) as zf:
        names = set(zf.namelist())
    assert names == {
        "data/tournaments/active_sampling_state_this.json",
        "data/matches/m-this.json",
        "data/challenge_bank/challenges/c-n.json",
        "data/defenses/d-x.json",
    }
    # An empty ./config packages no members (only the data arcs remain),
    # and an empty config dir is still a valid config dir.
    assert not any(n.startswith("config/") for n in names)


def test_offer_contribution_zip_confirm_and_decline(
    contributor_data: Path,
    contributor_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The end-of-flow prompt: confirming writes the zip (data slice + the
    ./config directory), declining (or Ctrl+C → None) only skips it."""
    from swe_duel.cli import run_tournament_active_sampling as m

    state_path = contributor_data / "tournaments" / "active_sampling_state_this.json"
    payload = json.loads(state_path.read_text())

    class _Confirm:
        def __init__(self, answer: object) -> None:
            self.answer = answer

        def ask(self) -> object:
            return self.answer

    monkeypatch.setattr(m.questionary, "confirm", lambda *a, **k: _Confirm(None))
    m._offer_contribution_zip(
        contributor_data, state_path, payload, [], tmp_path, config_dir=contributor_config
    )
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".zip"]
    assert "contribution zip skipped" in capsys.readouterr().out

    monkeypatch.setattr(m.questionary, "confirm", lambda *a, **k: _Confirm(True))
    m._offer_contribution_zip(
        contributor_data, state_path, payload, [], tmp_path, config_dir=contributor_config
    )
    zips = [p for p in tmp_path.iterdir() if p.suffix == ".zip"]
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as zf:
        assert "data/tournaments/active_sampling_state_this.json" in zf.namelist()
        assert "config/models.yaml" in zf.namelist()


def test_prepared_zip_round_trips_through_the_importer(
    contributor_data: Path,
    contributor_config: Path,
    tmp_path: Path,
) -> None:
    """The zip the AS console prepares is exactly what
    swe-duel-tournament-update accepts: safe extraction under a ``data/``
    root, single-tournament resolution over the carried states, and the
    turn-referenced challenge/defense payloads its merge consumes — while
    the config/ members ride along untouched (the import reads only the
    data subtree)."""
    from swe_duel.cli.tournament_update import (
        _Identity,
        _extract_submission,
        _load_as_states,
        _load_scope_payloads,
        _resolve_tournament,
    )

    state_path = contributor_data / "tournaments" / "active_sampling_state_this.json"
    target = tmp_path / "sub.zip"
    _prepare_contribution_zip(
        contributor_data,
        state_path,
        json.loads(state_path.read_text()),
        ["m-this"],
        tmp_path / "submissions",
        zip_path=target,
        config_dir=contributor_config,
    )

    dest = tmp_path / "dest"
    dest.mkdir()
    data_root, total = _extract_submission(target, dest)
    assert data_root == dest / "data"
    assert total > 0
    # The config/ subtree never confuses the data-prefix locator and is
    # never extracted — it stays provenance inside the zip.
    assert not (dest / "config").exists()
    with zipfile.ZipFile(target) as zf:
        assert "config/repos/flask.yaml" in zf.namelist()

    tid, selected = _resolve_tournament(_load_as_states(data_root / "tournaments"), None)
    assert tid == _ZIP_TID
    assert {p.name for p, _s in selected} == {
        "active_sampling_state_this.json",
        "active_sampling_state_earlier.json",
    }
    field = {_Identity.from_cid(cid) for _p, s in selected for cid in s["field"]}
    challenges, defenses = _load_scope_payloads(data_root, field)
    assert {c["challenge_id"] for c in challenges} == {"c-n", "f-n"}
    assert {d["defense_id"] for d in defenses} == {"d-a", "d-b"}


# ── --resume-state helpers ─────────────────────────────────────


def test_resolve_resume_state_path_accepts_id_stem_and_path(
    tmp_path: Path,
) -> None:
    """The state locator accepts the bare id the console printed, the full
    file stem/name, or a direct path; an unknown id fails fast listing the
    available states."""
    from swe_duel.cli import run_tournament_active_sampling as m

    tdir = tmp_path / "tournaments"
    state = _write_json(
        tdir / "active_sampling_state_39e956d4.json",
        {"format": "active_sampling"},
    )
    assert m._resolve_resume_state_path("39e956d4", tdir) == state
    assert m._resolve_resume_state_path("active_sampling_state_39e956d4", tdir) == state
    assert m._resolve_resume_state_path("active_sampling_state_39e956d4.json", tdir) == state
    assert m._resolve_resume_state_path(str(state), tdir) == state
    with pytest.raises(SystemExit, match="39e956d4"):
        m._resolve_resume_state_path("missing-id", tdir)
    with pytest.raises(SystemExit, match="no active_sampling_state"):
        m._resolve_resume_state_path("missing-id", tmp_path / "empty")


def test_load_resume_plan_replays_state_record(tmp_path: Path) -> None:
    """The plan rebuilds the state's entrants, pairings, and recorded arena
    from the persisted payload — the resume path re-drives exactly what the
    earlier session sampled."""
    from swe_duel.cli import run_tournament_active_sampling as m
    from swe_duel.engine.swiss import SwissPairing

    tdir = tmp_path / "tournaments"
    _write_json(
        tdir / "active_sampling_state_this.json", _zip_state_payload("this", _ZIP_TID)
    )
    plan = m._load_resume_plan("this", tdir)
    assert plan.state_path == tdir / "active_sampling_state_this.json"
    assert plan.newcomers == [_ZIP_N]
    assert plan.pairings == [SwissPairing(model_a=_ZIP_N, model_b=_ZIP_A)]
    assert plan.repo_names == ["flask"]
    assert plan.targets_per_repo == 1
    assert plan.state_payload["status"] == "complete"


def test_load_resume_plan_rejects_unusable_states(tmp_path: Path) -> None:
    """Only a state that recorded entrants, sampled pairings, and its arena
    can be resumed — anything else would re-drive the wrong matchups."""
    from swe_duel.cli import run_tournament_active_sampling as m

    tdir = tmp_path / "tournaments"
    _write_json(
        tdir / "active_sampling_state_a.json", {"format": "round_robin"}
    )
    with pytest.raises(SystemExit, match="not an active-sampling state"):
        m._load_resume_plan("a", tdir)

    no_entrants = _zip_state_payload("b", _ZIP_TID)
    no_entrants["newcomers"] = []
    no_entrants["targets"] = []
    no_entrants["pairings"] = []
    _write_json(tdir / "active_sampling_state_b.json", no_entrants)
    with pytest.raises(SystemExit, match="no new entrants"):
        m._load_resume_plan("b", tdir)

    no_pairings = _zip_state_payload("c", _ZIP_TID)
    no_pairings["pairings"] = []
    _write_json(tdir / "active_sampling_state_c.json", no_pairings)
    with pytest.raises(SystemExit, match="no sampled pairings"):
        m._load_resume_plan("c", tdir)

    no_repos = _zip_state_payload("d", _ZIP_TID)
    no_repos["repo_names"] = []
    _write_json(tdir / "active_sampling_state_d.json", no_repos)
    with pytest.raises(SystemExit, match="no repo_names"):
        m._load_resume_plan("d", tdir)


def test_resolve_resume_rankings_matches_state_record(
    tmp_path: Path, rankings_file: Path
) -> None:
    """A resumed state defaults to the rankings export it entered; explicit
    --tournament/--rankings must agree with that record (a different
    export means a different incumbent field)."""
    from swe_duel.cli import run_tournament_active_sampling as m

    tdir = tmp_path / "tournaments"
    rk = tmp_path / "rankings"
    _write_json(
        tdir / "active_sampling_state_this.json",
        _zip_state_payload("this", None, source=str(rankings_file)),
    )
    plan = m._load_resume_plan("this", tdir)

    # No flags: the state's recorded export is the default.
    args = argparse.Namespace(tournament=None, rankings=None)
    assert m._resolve_resume_rankings(args, plan, rk) == rankings_file
    # An explicit --rankings agreeing with the record is accepted.
    args.rankings = str(rankings_file)
    assert m._resolve_resume_rankings(args, plan, rk) == rankings_file
    # A different export fails fast — never resume against another field.
    args.rankings = str(tmp_path / "other.json")
    with pytest.raises(SystemExit, match="was sampled from"):
        m._resolve_resume_rankings(args, plan, rk)
    # Mutually exclusive flags fail as in the fresh flow.
    with pytest.raises(SystemExit, match="mutually exclusive"):
        m._resolve_resume_rankings(
            argparse.Namespace(tournament="tid", rankings=str(rankings_file)),
            plan,
            rk,
        )

    # An entered-tournament state: its rankings_<id>.json resolves, a
    # disagreeing --tournament fails fast, a missing export asks for a flag.
    _write_json(tdir / "active_sampling_state_ent.json", _zip_state_payload("ent", _ZIP_TID))
    plan_ent = m._load_resume_plan("ent", tdir)
    ent_export = _write_json(
        rk / f"rankings_{_ZIP_TID}.json",
        {"rankings": [{"model": "a", "harness": "mini-swe-agent"}]},
    )
    assert (
        m._resolve_resume_rankings(
            argparse.Namespace(tournament=None, rankings=None), plan_ent, rk
        )
        == ent_export
    )
    with pytest.raises(SystemExit, match="entered tournament"):
        m._resolve_resume_rankings(
            argparse.Namespace(tournament="tid-9999", rankings=None), plan_ent, rk
        )
    ent_export.unlink()
    with pytest.raises(SystemExit, match="pass --rankings"):
        m._resolve_resume_rankings(
            argparse.Namespace(tournament=None, rankings=None), plan_ent, rk
        )

    # A state that records neither an entered tournament nor a usable
    # export path cannot be placed and needs an explicit flag.
    _write_json(
        tdir / "active_sampling_state_lost.json",
        _zip_state_payload("lost", None, source=""),
    )
    plan_lost = m._load_resume_plan("lost", tdir)
    with pytest.raises(SystemExit, match="pass --tournament <id> or --rankings"):
        m._resolve_resume_rankings(
            argparse.Namespace(tournament=None, rankings=None), plan_lost, rk
        )
