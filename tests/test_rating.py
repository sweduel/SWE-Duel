"""Phase 12: Rating systems."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from swe_duel.models import (
    AgentTrajectory,
    BlueFix,
    DefenseResult,
    MatchOutcome,
    MatchResult,
    TurnResult,
    TurnScore,
)
from swe_duel.scoring.rating import (
    BradleyTerryRating,
    EloRating,
    TrueSkillRating,
    compute_all_ratings,
)


# ── helpers ──────────────────────────────────────────────


def _round(
    red_id: str, blue_id: str, red_score: float, index: int = 0
) -> TurnResult:
    sc = TurnScore(
        s_regression=1.0,
        s_feature=1.0 - red_score,
        s_bugfix=1.0 - red_score,
        blue_composite=1.0 - red_score,
        red_composite=red_score,
        test_details={},
    )
    traj = AgentTrajectory(
        steps=[], total_steps=0, total_input_tokens=0,
        total_output_tokens=0, total_cost_usd=0.0,
        model_id=blue_id, duration_seconds=0.0,
    )
    defense = DefenseResult(
        defense_id=str(uuid.uuid4()),
        challenge_id=str(uuid.uuid4()),
        blue_model_id=blue_id,
        blue_fix=BlueFix(
            review_findings=[], fix_explanation="", fix_diff="",
            modified_file_contents={}, agent_trajectory=traj,
        ),
        score=sc,
        duration_seconds=0.0,
        cost_usd=0.0,
        timestamp=datetime.now(timezone.utc),
    )
    from swe_duel.models import ChallengeRecord, GateResult, GateStatus, RedChallenge, RedValidationResult
    challenge = RedChallenge(
        target_files=["x.py"], exploration_summary="", feature_spec="",
        feature_rationale="", pr_diff="", modified_file_contents={},
        original_file_contents={}, feature_test_code="",
        bug_type="loop bound skips the final element", bug_description="", bug_location="",
        bug_test_code="", agent_trajectory=traj,
    )
    rec = ChallengeRecord(
        challenge_id=defense.challenge_id,
        red_model_id=red_id,
        repo_name="r",
        repo_commit_sha="c",
        target_files=["x.py"],
        challenge=challenge,
        validation=RedValidationResult(
            passed=True,
            gate_results=[GateResult("g", GateStatus.PASSED, "ok")],
            attempt_number=1,
        ),
        generated_at=datetime.now(timezone.utc),
        generation_cost_usd=0.0,
        generation_retries=0,
    )
    return TurnResult(
        turn_id=str(uuid.uuid4()),
        turn_index=index,
        challenge_record=rec,
        defense_result=defense,
        red_model_id=red_id,
        blue_model_id=blue_id,
    )


def _match(a: str, b: str, outcome: MatchOutcome, turns=None) -> MatchResult:
    return MatchResult(
        match_id=str(uuid.uuid4()),
        model_a_id=a, model_b_id=b, repo_name="r",
        turns=turns or [],
        model_a_total=0.0, model_b_total=0.0,
        outcome=outcome,
        duration_seconds=0.0,
        total_cost_usd=0.0,
        timestamp=datetime.now(timezone.utc),
    )


# ── ELO ────────────────────────────────────────────────


def test_elo_initial(record):
    elo = EloRating()
    elo._ensure("X")
    record("rating_X", elo.ratings["X"])
    assert elo.ratings["X"] == 1500.0


def test_elo_update_a_wins(record):
    elo = EloRating()
    elo.update("A", "B", MatchOutcome.MODEL_A_WINS)
    record("A", elo.ratings["A"])
    record("B", elo.ratings["B"])
    assert elo.ratings["A"] > 1500.0
    assert elo.ratings["B"] < 1500.0


def test_elo_update_draw(record):
    elo = EloRating()
    elo.ratings["A"] = 1600.0
    elo.ratings["B"] = 1400.0
    elo.update("A", "B", MatchOutcome.DRAW)
    record("A", elo.ratings["A"])
    record("B", elo.ratings["B"])
    assert elo.ratings["A"] < 1600.0
    assert elo.ratings["B"] > 1400.0


def test_elo_expected_score(record):
    e_eq = EloRating.expected_score(1500, 1500)
    e_gap = EloRating.expected_score(1900, 1500)
    record("equal", e_eq)
    record("gap_400", e_gap)
    assert e_eq == pytest.approx(0.5, abs=1e-9)
    assert e_gap == pytest.approx(0.909, abs=0.01)


def test_elo_symmetric_updates(record):
    elo = EloRating()
    before = 2 * 1500.0
    elo.update("A", "B", MatchOutcome.MODEL_A_WINS)
    total = elo.ratings["A"] + elo.ratings["B"]
    record("before", before)
    record("after", total)
    assert total == pytest.approx(before, abs=1e-6)


# ── Bradley-Terry ──────────────────────────────────────


def test_bradley_terry_two_models(record):
    bt = BradleyTerryRating()
    matches = [_match("A", "B", MatchOutcome.MODEL_A_WINS) for _ in range(8)]
    matches += [_match("A", "B", MatchOutcome.MODEL_B_WINS) for _ in range(2)]
    strengths = bt.fit(matches)
    record("strengths", strengths)
    assert strengths["A"] > strengths["B"]


def test_bradley_terry_three_models(record):
    bt = BradleyTerryRating()
    matches = []
    matches += [_match("A", "B", MatchOutcome.MODEL_A_WINS) for _ in range(7)]
    matches += [_match("A", "B", MatchOutcome.MODEL_B_WINS) for _ in range(3)]
    matches += [_match("B", "C", MatchOutcome.MODEL_A_WINS) for _ in range(7)]
    matches += [_match("B", "C", MatchOutcome.MODEL_B_WINS) for _ in range(3)]
    matches += [_match("A", "C", MatchOutcome.MODEL_A_WINS) for _ in range(9)]
    matches += [_match("A", "C", MatchOutcome.MODEL_B_WINS) for _ in range(1)]
    strengths = bt.fit(matches)
    record("strengths", strengths)
    assert strengths["A"] > strengths["B"] > strengths["C"]


# ── TrueSkill ──────────────────────────────────────────


def test_trueskill_convergence(record):
    ts = TrueSkillRating()
    ts._ensure("A")
    ts._ensure("B")
    initial_sigma = ts.ratings["A"].sigma
    for _ in range(20):
        ts.update("A", "B", MatchOutcome.MODEL_A_WINS)
    final_sigma = ts.ratings["A"].sigma
    record("initial_sigma", initial_sigma)
    record("final_sigma", final_sigma)
    assert final_sigma < initial_sigma


# ── Unified ────────────────────────────────────────────


def test_compute_all_ratings(record):
    import random as _r
    rng = _r.Random(0)
    matches = []
    for _ in range(10):
        a, b = rng.sample(["A", "B", "C"], 2)
        outcome = rng.choice(
            [MatchOutcome.MODEL_A_WINS, MatchOutcome.MODEL_B_WINS, MatchOutcome.DRAW]
        )
        matches.append(_match(a, b, outcome))
    snapshots = compute_all_ratings(matches)
    record("models", sorted(snapshots.keys()))
    record("ratings", {m: s.elo for m, s in snapshots.items()})
    assert set(snapshots.keys()) == {"A", "B", "C"}
    for snap in snapshots.values():
        assert snap.matches_played > 0
        assert snap.trueskill_sigma > 0


def test_per_role_elo(record):
    # A strong Red, weak Blue. B the opposite.
    #   A attacks B: red_score=0.9 (A wins as Red)    → A.red_elo ↑
    #   B attacks A: red_score=0.1 (A wins as Blue too) → in this match A wins both roles.
    # We want asymmetry, so: A wins when Red, loses when Blue.
    #   A attacks B: red_score=0.9 → A.red_elo ↑, B.blue_elo ↓
    #   B attacks A: red_score=0.9 → B.red_elo ↑, A.blue_elo ↓
    # But that is symmetric for the pair's ROLE elos. Use third model C for mismatch.
    turns = []
    for i in range(6):
        turns.append(_round("A", "C", red_score=0.95, index=2 * i))
        turns.append(_round("C", "A", red_score=0.95, index=2 * i + 1))
    match_ac = _match("A", "C", MatchOutcome.DRAW, turns=turns)
    # B: opposite profile — weak Red, strong Blue vs C
    turns_b = []
    for i in range(6):
        turns_b.append(_round("B", "C", red_score=0.05, index=2 * i))
        turns_b.append(_round("C", "B", red_score=0.05, index=2 * i + 1))
    match_bc = _match("B", "C", MatchOutcome.DRAW, turns=turns_b)
    snapshots = compute_all_ratings([match_ac, match_bc])
    record("A_red", snapshots["A"].red_elo)
    record("A_blue", snapshots["A"].blue_elo)
    record("B_red", snapshots["B"].red_elo)
    record("B_blue", snapshots["B"].blue_elo)
    # A excels at Red (high red_score when attacking)
    assert snapshots["A"].red_elo > snapshots["A"].blue_elo
    # B excels at Blue (low red_score when B defends → high blue score)
    assert snapshots["B"].blue_elo > snapshots["B"].red_elo


def test_compute_all_ratings_seeds_roster(record):
    snaps = compute_all_ratings(
        [_match("A", "B", MatchOutcome.MODEL_A_WINS)],
        model_ids=["A", "B", "C"],
    )
    record("models", sorted(snaps))
    assert set(snaps) == {"A", "B", "C"}
    assert snaps["C"].matches_played == 0
    assert snaps["C"].elo == 1500.0
