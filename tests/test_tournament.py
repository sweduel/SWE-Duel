"""Phase 12: TournamentOrchestrator tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.config import (
    ArenaConfig,
    ChallengeBankConfig,
    MatchConfig,
    ModelConfig,
    RepoConfig,
)
from swe_duel.engine.tournament import TournamentOrchestrator
from swe_duel.logging.artifacts import ArtifactLogger
from swe_duel.models import (
    AgentTrajectory,
    BugType,
    ChallengeRecord,
    GateResult,
    GateStatus,
    MatchOutcome,
    MatchResult,
    RedChallenge,
    RedValidationResult,
)


# ── helpers ──────────────────────────────────────────────


def _trajectory() -> AgentTrajectory:
    return AgentTrajectory(
        steps=[], total_steps=0, total_input_tokens=0,
        total_output_tokens=0, total_cost_usd=0.0,
        model_id="mock", duration_seconds=0.0,
    )


def _challenge_record(red_model: str, repo: str) -> ChallengeRecord:
    ch = RedChallenge(
        target_files=["f.py"], exploration_summary="", feature_spec="",
        feature_rationale="", pr_diff="", modified_file_contents={},
        original_file_contents={}, feature_test_code="",
        bug_type=BugType.LOGIC_ERROR, bug_description="", bug_location="",
        bug_test_code="", agent_trajectory=_trajectory(),
    )
    return ChallengeRecord(
        challenge_id=str(uuid.uuid4()),
        red_model_id=red_model, repo_name=repo, repo_commit_sha="abc",
        target_files=["f.py"], challenge=ch,
        validation=RedValidationResult(
            passed=True,
            gate_results=[GateResult("g", GateStatus.PASSED, "ok")],
            attempt_number=1,
        ),
        generated_at=datetime.now(timezone.utc),
        generation_cost_usd=0.01, generation_retries=0,
    )


def _repo(name: str) -> RepoConfig:
    return RepoConfig(
        name=name, url=f"local://{name}", commit="abc",
        language="python", test_command="pytest", docker_image="img",
    )


def _model(model_id: str) -> ModelConfig:
    return ModelConfig(model_id=model_id)


def _match_result(a: str, b: str, repo: str, outcome: MatchOutcome) -> MatchResult:
    return MatchResult(
        match_id=str(uuid.uuid4()), model_a_id=a, model_b_id=b, repo_name=repo,
        turns=[], model_a_total=1.0 if outcome == MatchOutcome.MODEL_A_WINS else 0.0,
        model_b_total=1.0 if outcome == MatchOutcome.MODEL_B_WINS else 0.0,
        outcome=outcome, duration_seconds=0.0, total_cost_usd=0.02,
        timestamp=datetime.now(timezone.utc),
    )


def _make_orchestrator(tmp_path: Path, *, models, repos, matches_per_pair=1, seed=42):
    store = ChallengeStore(bank_dir=tmp_path / "bank")
    logger = ArtifactLogger(data_dir=tmp_path / "data")
    config = ArenaConfig(
        match=MatchConfig(turns_per_player=2),
        challenge_bank=ChallengeBankConfig(target_challenges_per_model_repo=2),
    )
    model_configs = {m.model_id.split("/")[-1]: m for m in models}
    repo_configs = {r.name: r for r in repos}
    return TournamentOrchestrator(
        model_configs=model_configs,
        repo_configs=repo_configs,
        challenge_store=store,
        workspace_manager=MagicMock(),
        config=config,
        artifact_logger=logger,
        matches_per_pair_per_repo=matches_per_pair,
        seed=seed,
    ), store, logger


# ── tests ────────────────────────────────────────────────


def test_generate_schedule(tmp_path, record):
    models = [_model("v/a"), _model("v/b"), _model("v/c")]
    repos = [_repo("r1"), _repo("r2")]
    orch, _, _ = _make_orchestrator(tmp_path, models=models, repos=repos)
    schedule = orch._generate_schedule()
    record("schedule", schedule)
    # 3 models → 3 pairs × 2 repos = 6 entries
    assert len(schedule) == 6
    # all pairs present per repo
    for repo_name in ["r1", "r2"]:
        pairs = sorted(tuple(sorted((a, b))) for a, b, r in schedule if r == repo_name)
        assert pairs == sorted([("v/a", "v/b"), ("v/a", "v/c"), ("v/b", "v/c")])


def test_generation_phase_populates_bank(tmp_path, record):
    models = [_model("v/a")]
    repos = [_repo("r1")]
    orch, store, _ = _make_orchestrator(tmp_path, models=models, repos=repos)

    generator = MagicMock()

    def fake_generate_pool(red_agent, repo_config, target_count, red_model_id):
        for _ in range(target_count):
            store.store(_challenge_record(red_model_id, repo_config.name))
        return store.pool_stats(red_model_id, repo_config.name)

    generator.generate_pool.side_effect = fake_generate_pool
    orch.challenge_generator = generator
    orch.red_agents = {"a": MagicMock()}

    stats = orch._run_generation_phase()
    record("keys", list(stats.keys()))
    record("total", stats["v/a::r1"].total_challenges)
    assert stats["v/a::r1"].total_challenges == 2
    assert generator.generate_pool.called


def test_generation_phase_skips_cached(tmp_path, record):
    models = [_model("v/a")]
    repos = [_repo("r1")]
    orch, store, _ = _make_orchestrator(tmp_path, models=models, repos=repos)
    # Pre-populate with enough challenges
    for _ in range(3):
        store.store(_challenge_record("v/a", "r1"))

    generator = MagicMock()
    orch.challenge_generator = generator
    orch.red_agents = {"a": MagicMock()}

    stats = orch._run_generation_phase()
    record("calls", generator.generate_pool.call_count)
    record("total", stats["v/a::r1"].total_challenges)
    assert generator.generate_pool.call_count == 0
    assert stats["v/a::r1"].total_challenges == 3


def test_evaluation_phase_runs_matches(tmp_path, record):
    models = [_model("v/a"), _model("v/b")]
    repos = [_repo("r1")]
    orch, _, _ = _make_orchestrator(tmp_path, models=models, repos=repos)

    mock_mo = MagicMock()
    mock_mo.execute.side_effect = lambda model_a_id, model_b_id, repo_config, seed: _match_result(
        model_a_id, model_b_id, repo_config.name, MatchOutcome.MODEL_A_WINS
    )
    orch.match_orchestrator = mock_mo

    matches = orch._run_evaluation_phase()
    record("n_matches", len(matches))
    record("execute_calls", mock_mo.execute.call_count)
    assert mock_mo.execute.call_count == 1  # 2 models, 1 pair × 1 repo
    assert len(matches) == 1


def test_checkpoint_and_resume(tmp_path, record):
    models = [_model("v/a"), _model("v/b"), _model("v/c")]
    repos = [_repo("r1")]
    orch, _, _ = _make_orchestrator(tmp_path, models=models, repos=repos)

    call_count = {"n": 0}

    def execute_fn(model_a_id, model_b_id, repo_config, seed):
        call_count["n"] += 1
        if call_count["n"] > 2:
            raise RuntimeError("simulated crash after 2 matches")
        return _match_result(model_a_id, model_b_id, repo_config.name, MatchOutcome.DRAW)

    mock_mo = MagicMock()
    mock_mo.execute.side_effect = execute_fn
    orch.match_orchestrator = mock_mo

    with pytest.raises(RuntimeError):
        orch._run_evaluation_phase()
    record("first_run_calls", call_count["n"])
    record("checkpoint_exists", orch.checkpoint_path.exists())
    assert orch.checkpoint_path.exists()

    # Resume: replace executor to succeed
    call_count["n"] = 0
    mock_mo.execute.side_effect = lambda model_a_id, model_b_id, repo_config, seed: _match_result(
        model_a_id, model_b_id, repo_config.name, MatchOutcome.MODEL_A_WINS
    )
    # Manually simulate resume logic on remaining
    import json
    data = json.loads(orch.checkpoint_path.read_text())
    completed_keys = set(data["completed_keys"])
    remaining = [t for t in orch._generate_schedule() if orch._schedule_key(t) not in completed_keys]
    record("remaining", remaining)
    record("completed_so_far", sorted(completed_keys))
    assert len(completed_keys) == 2
    assert len(remaining) == 1


def test_end_to_end_mini_tournament(tmp_path, record):
    models = [_model("v/a"), _model("v/b")]
    repos = [_repo("r1")]
    orch, store, logger = _make_orchestrator(tmp_path, models=models, repos=repos)

    for _ in range(2):
        store.store(_challenge_record("v/a", "r1"))
        store.store(_challenge_record("v/b", "r1"))

    mock_mo = MagicMock()
    mock_mo.execute.side_effect = lambda model_a_id, model_b_id, repo_config, seed: _match_result(
        model_a_id, model_b_id, repo_config.name, MatchOutcome.MODEL_A_WINS
    )
    orch.match_orchestrator = mock_mo

    result = orch.execute()
    record("n_matches", len(result.matches))
    record("n_ratings", len(result.final_ratings))
    record("head_to_head_keys", sorted(result.head_to_head.keys()))
    record("total_eval_cost", result.total_evaluation_cost_usd)

    assert len(result.matches) == 1
    assert {r.model_id for r in result.final_ratings} == {"v/a", "v/b"}
    assert "v/a" in result.head_to_head and "v/b" in result.head_to_head["v/a"]
    assert result.total_evaluation_cost_usd == pytest.approx(0.02)
