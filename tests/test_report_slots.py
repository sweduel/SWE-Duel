"""Tests for the multi-slot reporting fixes — no LLM, no Docker.

Covers:
  * ``ArtifactLogger.log_match`` persists a per-turn ``repo_name`` (so the
    report can group every turn — auto-win turns included — under its real
    repo).
  * ``build_report._classify_turns`` prefers the per-turn ``repo_name`` and no
    longer collapses multiple auto-win turns into a shared ``(auto-win)`` bucket
    (the cause of phantom extra "Turn N" rows).
  * ``build_report._aggregate_usage`` folds failed-only-slot generation cost
    into the per-competitor cost columns (scoped to the tournament), so the
    cost columns cover all attempts under the configured target count.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


from swe_duel.logging.artifacts import ArtifactLogger  # noqa: E402
from swe_duel.models import (  # noqa: E402
    AgentTrajectory,
    BlueFix,
    ChallengeRecord,
    DefenseResult,
    MatchOutcome,
    MatchResult,
    RedChallenge,
    RedValidationResult,
    TurnResult,
    TurnScore,
)
from swe_duel.cli import build_report  # noqa: E402


# ── builders ───────────────────────────────────────────────


def _traj() -> AgentTrajectory:
    return AgentTrajectory(
        steps=[], total_steps=0, total_input_tokens=0, total_output_tokens=0,
        total_cost_usd=0.0, model_id="m", duration_seconds=0.0,
    )


def _challenge_record(cid: str, red: str, repo: str) -> ChallengeRecord:
    challenge = RedChallenge(
        target_files=[], exploration_summary="", feature_spec="",
        feature_rationale="", pr_diff="", modified_file_contents={},
        original_file_contents={}, feature_test_code="", bug_type=None,
        bug_description=None, bug_location=None, bug_test_code=None,
        agent_trajectory=_traj(),
    )
    return ChallengeRecord(
        challenge_id=cid, red_model_id=red, repo_name=repo, repo_commit_sha="",
        target_files=[], challenge=challenge,
        validation=RedValidationResult(passed=True, gate_results=[], attempt_number=1),
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        generation_cost_usd=0.0, generation_retries=0,
    )


def _defense(did: str, blue: str, blue_comp: float, auto: bool = False) -> DefenseResult:
    return DefenseResult(
        defense_id=did, challenge_id="", blue_model_id=blue,
        blue_fix=BlueFix(
            review_findings=[], fix_explanation="", fix_diff="",
            modified_file_contents={}, agent_trajectory=_traj(),
        ),
        score=TurnScore(
            s_regression=1.0, s_feature=1.0, s_bugfix=1.0,
            blue_composite=blue_comp, red_composite=1.0 - blue_comp,
            test_details={"auto_win": True} if auto else {},
        ),
        duration_seconds=0.0, cost_usd=0.0,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _turn(cid: str, did: str, red: str, blue: str, repo: str, auto: bool = False) -> TurnResult:
    return TurnResult(
        turn_id=str(uuid.uuid4()), turn_index=0,
        challenge_record=_challenge_record(cid, red, repo),
        defense_result=_defense(did, blue, 0.0 if auto else 1.0, auto=auto),
        red_model_id=red, blue_model_id=blue,
    )


# ── log_match persists per-turn repo_name ───────────────────


def test_log_match_persists_per_turn_repo_name(tmp_path, record):
    logger = ArtifactLogger(data_dir=tmp_path)
    # Two auto-win turns for the SAME side but DIFFERENT repos — the case that
    # previously rendered a phantom "Turn 2".
    turns = [
        _turn("", "d1", "A", "B", "helmet", auto=True),
        _turn("", "d2", "A", "B", "cjson", auto=True),
        _turn("c1", "d3", "A", "B", "flask"),
    ]
    result = MatchResult(
        match_id="m1", model_a_id="A", model_b_id="B",
        repo_name="helmet,cjson,flask", turns=turns,
        model_a_total=0.0, model_b_total=3.0, outcome=MatchOutcome.MODEL_B_WINS,
        duration_seconds=0.0, total_cost_usd=0.0,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    path = logger.log_match(result)
    import json

    payload = json.loads(path.read_text())
    repos = [t["repo_name"] for t in payload["turns"]]
    record("per_turn_repos", repos)
    assert repos == ["helmet", "cjson", "flask"]


# ── _classify_turns no longer collapses auto-wins ───────────


def test_classify_turns_uses_per_turn_repo_no_phantom_slots(record):
    # A match with two auto-win turns (same side A-attacks) in different repos.
    m = {
        "model_a_id": "A",
        "model_b_id": "B",
        "turns": [
            {"red_model_id": "A", "blue_model_id": "B", "challenge_id": "",
             "repo_name": "helmet", "score": {"test_details": {"auto_win": True}}},
            {"red_model_id": "A", "blue_model_id": "B", "challenge_id": "",
             "repo_name": "cjson", "score": {"test_details": {"auto_win": True}}},
        ],
    }
    classified = build_report._classify_turns(m, {})
    slots = [slot for slot, _side, _repo, _t in classified]
    repos = [repo for _slot, _side, repo, _t in classified]
    record("slots", slots)
    record("repos", repos)
    # Both turns are slot 0 of their own (repo, side) group — NOT 0 and 1.
    assert slots == [0, 0]
    assert repos == ["helmet", "cjson"]
    assert "(auto-win)" not in repos


def test_classify_turns_falls_back_to_challenge_index(record):
    # Old un-migrated turn (no repo_name) still resolves via the challenge index.
    m = {
        "model_a_id": "A", "model_b_id": "B",
        "turns": [
            {"red_model_id": "A", "blue_model_id": "B", "challenge_id": "c1",
             "score": {"test_details": {}}},
        ],
    }
    chal_idx = {"c1": {"repo": "jwt"}}
    classified = build_report._classify_turns(m, chal_idx)
    record("repo", classified[0][2])
    assert classified[0][2] == "jwt"


# ── _aggregate_usage folds in failed-only slots ─────────────


def test_aggregate_usage_includes_failed_only_slot_cost(record):
    # One real success turn (A on flask) + matches scoping A,B over flask+helmet.
    matches = [{
        "model_a_id": "A#h", "model_b_id": "B#h", "repo_name": "flask,helmet",
        "turns": [
            {"red_model_id": "A#h", "blue_model_id": "B#h",
             "challenge_id": "cok", "defense_id": "dok",
             "repo_name": "flask", "score": {"test_details": {}}},
            # auto-win: A had no helmet challenge (failed-only slot).
            {"red_model_id": "A#h", "blue_model_id": "B#h",
             "challenge_id": "", "defense_id": "dauto",
             "repo_name": "helmet", "score": {"test_details": {"auto_win": True}}},
        ],
    }]
    chal_idx = {
        "cok": {"red_model": "A", "red_composite": "A#h", "repo": "flask",
                "slot": 1, "gen_cost": 2.0, "val_cost": 0.5,
                "in_tokens": 100, "out_tokens": 50},
    }
    def_idx = {
        "dok": {"blue_model": "B", "eval_cost": 1.0, "in_tokens": 10, "out_tokens": 5},
    }
    failed_idx = {
        # A's failed-only helmet slot (no success there) — should be added.
        "f1": {"red_composite": "A#h", "repo": "helmet", "slot": 1,
               "gen_cost": 3.0, "in_tokens": 200, "out_tokens": 80},
        # A failed attempt in flask slot 1 that DID eventually succeed — must
        # NOT be double-counted (its cost is already in cok.gen_cost).
        "f2": {"red_composite": "A#h", "repo": "flask", "slot": 1,
               "gen_cost": 9.9, "in_tokens": 999, "out_tokens": 999},
        # Out-of-scope repo — ignored.
        "f3": {"red_composite": "A#h", "repo": "sqlalchemy", "slot": 1,
               "gen_cost": 5.0, "in_tokens": 1, "out_tokens": 1},
        # Out-of-scope competitor — ignored.
        "f4": {"red_composite": "Z#h", "repo": "helmet", "slot": 1,
               "gen_cost": 7.0, "in_tokens": 1, "out_tokens": 1},
    }
    per_model, totals = build_report._aggregate_usage(
        matches, chal_idx, def_idx, failed_idx
    )
    record("per_model", per_model)
    a = per_model["A#h"]
    # 2.0 (success, incl. its in-slot fails) + 3.0 (failed-only helmet slot).
    assert a["gen_cost"] == 5.0
    assert a["val_cost"] == 0.5
    assert a["in_tokens"] == 100 + 200  # success + failed-only slot
    b = per_model["B#h"]
    assert b["eval_cost"] == 1.0
    # f2 (succeeded slot), f3 (out-of-scope repo), f4 (out-of-scope comp) excluded.
    assert totals["gen_cost"] == 5.0


def test_aggregate_usage_without_failed_idx_unchanged(record):
    matches = [{
        "model_a_id": "A#h", "model_b_id": "B#h", "repo_name": "flask",
        "turns": [
            {"red_model_id": "A#h", "blue_model_id": "B#h",
             "challenge_id": "cok", "defense_id": "dok",
             "repo_name": "flask", "score": {"test_details": {}}},
        ],
    }]
    chal_idx = {"cok": {"red_model": "A", "red_composite": "A#h", "repo": "flask",
                        "slot": 1, "gen_cost": 2.0, "val_cost": 0.0,
                        "in_tokens": 0, "out_tokens": 0}}
    per_model, _ = build_report._aggregate_usage(matches, chal_idx, {}, None)
    record("gen_cost", per_model["A#h"]["gen_cost"])
    assert per_model["A#h"]["gen_cost"] == 2.0
