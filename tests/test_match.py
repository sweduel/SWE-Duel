"""Phase 11: MatchOrchestrator tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.config import ArenaConfig, MatchConfig, ModelConfig, RepoConfig
from swe_duel.engine.match import MatchOrchestrator
from swe_duel.logging.artifacts import ArtifactLogger
from swe_duel.models import (
    AgentTrajectory,
    BlueFix,
    ChallengeRecord,
    DefenseResult,
    GateResult,
    GateStatus,
    MatchOutcome,
    RedChallenge,
    RedValidationResult,
    TurnScore,
)


# ── Builders ──────────────────────────────────────────────


def _trajectory() -> AgentTrajectory:
    return AgentTrajectory(
        steps=[],
        total_steps=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.0,
        model_id="test/mock",
        duration_seconds=0.0,
    )


def _challenge() -> RedChallenge:
    return RedChallenge(
        target_files=["src/foo.py"],
        exploration_summary="e",
        feature_spec="s",
        feature_rationale="r",
        pr_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n x\n+y\n",
        modified_file_contents={"src/foo.py": "c"},
        original_file_contents={"src/foo.py": "o"},
        feature_test_code="def test_f():\n    assert True\n",
        bug_type="loop bound skips the final element",
        bug_description="d",
        bug_location="l",
        bug_test_code="def test_b():\n    assert True\n",
        agent_trajectory=_trajectory(),
    )


def _passing_validation() -> RedValidationResult:
    return RedValidationResult(
        passed=True,
        gate_results=[
            GateResult(
                gate_name="g",
                status=GateStatus.PASSED,
                message="ok",
            )
        ],
        attempt_number=1,
    )


def _challenge_record(
    red_model: str,
    repo: str,
    commit: str = "abc",
    harness: str = "mini-swe-agent",
    effort: str = "",
    provider: str = "",
) -> ChallengeRecord:
    return ChallengeRecord(
        challenge_id=str(uuid.uuid4()),
        red_model_id=red_model,
        repo_name=repo,
        repo_commit_sha=commit,
        target_files=["src/foo.py"],
        challenge=_challenge(),
        validation=_passing_validation(),
        generated_at=datetime.now(timezone.utc),
        generation_cost_usd=0.0,
        generation_retries=0,
        red_harness_id=harness,
        red_reasoning_effort=effort,
        red_provider=provider,
    )


def _repo_config(commit: str = "abc") -> RepoConfig:
    return RepoConfig(
        name="mock_repo",
        url="local://mock_repo",
        commit=commit,
        language="python",
        test_command="pytest -q",
        docker_image="swe-duel-mock",
    )


def _blue_fix(cost: float = 0.0) -> BlueFix:
    return BlueFix(
        review_findings=[],
        fix_explanation="x",
        fix_diff="",
        modified_file_contents={},
        agent_trajectory=AgentTrajectory(
            steps=[],
            total_steps=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=cost,
            model_id="blue",
            duration_seconds=0.0,
        ),
    )


def _defense(challenge_id: str, blue_model: str, blue_composite: float) -> DefenseResult:
    score = TurnScore(
        s_regression=1.0,
        s_feature=1.0,
        s_bugfix=blue_composite,
        blue_composite=blue_composite,
        red_composite=1.0 - blue_composite,
        test_details={},
    )
    return DefenseResult(
        defense_id=str(uuid.uuid4()),
        challenge_id=challenge_id,
        blue_model_id=blue_model,
        blue_fix=_blue_fix(),
        score=score,
        duration_seconds=0.01,
        cost_usd=0.0,
        timestamp=datetime.now(timezone.utc),
    )


def _populate_store(
    store: ChallengeStore,
    red_model: str,
    repo: str,
    count: int,
    *,
    harness: str = "mini-swe-agent",
    effort: str = "",
    provider: str = "",
):
    records = [
        _challenge_record(red_model, repo, harness=harness, effort=effort, provider=provider)
        for _ in range(count)
    ]
    for r in records:
        store.store(r)
    return records


def _make_orchestrator(
    tmp_path: Path,
    *,
    turns_per_player: int = 3,
    blue_composite_when_b_defends: float = 1.0,
    blue_composite_when_a_defends: float = 0.0,
    draw_margin: float = 0.25,
):
    store = ChallengeStore(bank_dir=tmp_path / "bank")
    logger = ArtifactLogger(data_dir=tmp_path / "data")
    config = ArenaConfig(match=MatchConfig(turns_per_player=turns_per_player))
    model_configs = {
        "A": ModelConfig(model_id="vendor/model-a"),
        "B": ModelConfig(model_id="vendor/model-b"),
    }

    def evaluator_factory(
        blue_model_id: str,
        blue_harness_id: str = "mini-swe-agent",
        blue_reasoning_effort: str = "",
        blue_provider: str = "",
    ):
        ev = MagicMock()

        def evaluate(repo_config, challenge_record, **kwargs):
            if blue_model_id == "vendor/model-b":
                blue_c = blue_composite_when_b_defends
            else:
                blue_c = blue_composite_when_a_defends
            return _defense(
                challenge_record.challenge_id, blue_model_id, blue_c
            )

        ev.evaluate.side_effect = evaluate
        return ev

    workspace_manager = MagicMock()
    orchestrator = MatchOrchestrator(
        model_configs=model_configs,
        challenge_store=store,
        workspace_manager=workspace_manager,
        config=config,
        artifact_logger=logger,
        defense_evaluator_factory=evaluator_factory,
        draw_margin=draw_margin,
    )
    return orchestrator, store


# ── Tests ─────────────────────────────────────────────────


def test_match_correct_sampling(tmp_path, record):
    orchestrator, store = _make_orchestrator(tmp_path, turns_per_player=3)
    _populate_store(store, "vendor/model-a", "mock_repo", 5)
    _populate_store(store, "vendor/model-b", "mock_repo", 5)

    result = orchestrator.execute(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=42
    )

    a_red_turns = [r for r in result.turns if r.red_model_id == "vendor/model-a"]
    b_red_turns = [r for r in result.turns if r.red_model_id == "vendor/model-b"]

    record("total_turns", len(result.turns))
    record("a_red_turns", len(a_red_turns))
    record("b_red_turns", len(b_red_turns))
    record("outcome", result.outcome.value)

    assert len(a_red_turns) == 3
    assert len(b_red_turns) == 3
    assert len(result.turns) == 6


def test_match_score_aggregation(tmp_path, record):
    # B defends A's challenges perfectly → blue_composite=1.0 for A-red turns
    # A defends B's challenges completely fails → blue_composite=0.0 for B-red turns
    orchestrator, store = _make_orchestrator(
        tmp_path,
        turns_per_player=2,
        blue_composite_when_b_defends=1.0,
        blue_composite_when_a_defends=0.0,
    )
    _populate_store(store, "vendor/model-a", "mock_repo", 3)
    _populate_store(store, "vendor/model-b", "mock_repo", 3)

    result = orchestrator.execute(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=7
    )

    # A-red turns (x2): red_composite=0.0 → A score contribution, B blue_composite=1.0 → B score contribution
    # B-red turns (x2): red_composite=1.0 → B score, A blue_composite=0.0 → A score
    # Expected: score_a = 0+0 (red) + 0+0 (blue) = 0.0
    #           score_b = 1+1 (red) + 1+1 (blue) = 4.0
    record("model_a_total", result.model_a_total)
    record("model_b_total", result.model_b_total)
    record("outcome", result.outcome.value)

    assert result.model_a_total == pytest.approx(0.0)
    assert result.model_b_total == pytest.approx(4.0)


def test_match_outcome_a_wins(tmp_path, record):
    orchestrator, store = _make_orchestrator(
        tmp_path,
        turns_per_player=2,
        blue_composite_when_b_defends=0.0,
        blue_composite_when_a_defends=1.0,
        draw_margin=0.25,
    )
    _populate_store(store, "vendor/model-a", "mock_repo", 3)
    _populate_store(store, "vendor/model-b", "mock_repo", 3)

    result = orchestrator.execute(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=1
    )

    record("outcome", result.outcome.value)
    record("model_a_total", result.model_a_total)
    record("model_b_total", result.model_b_total)

    assert result.outcome == MatchOutcome.MODEL_A_WINS
    assert result.model_a_total > result.model_b_total


def test_match_outcome_draw(tmp_path, record):
    orchestrator, store = _make_orchestrator(
        tmp_path,
        turns_per_player=2,
        blue_composite_when_b_defends=0.5,
        blue_composite_when_a_defends=0.5,
        draw_margin=0.25,
    )
    _populate_store(store, "vendor/model-a", "mock_repo", 3)
    _populate_store(store, "vendor/model-b", "mock_repo", 3)

    result = orchestrator.execute(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=3
    )

    record("outcome", result.outcome.value)
    record("model_a_total", result.model_a_total)
    record("model_b_total", result.model_b_total)

    assert result.outcome == MatchOutcome.DRAW
    assert abs(result.model_a_total - result.model_b_total) <= 0.25


def test_match_outcome_b_wins(tmp_path, record):
    orchestrator, store = _make_orchestrator(
        tmp_path,
        turns_per_player=2,
        blue_composite_when_b_defends=1.0,
        blue_composite_when_a_defends=0.0,
        draw_margin=0.25,
    )
    _populate_store(store, "vendor/model-a", "mock_repo", 3)
    _populate_store(store, "vendor/model-b", "mock_repo", 3)

    result = orchestrator.execute(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=5
    )

    record("outcome", result.outcome.value)
    record("model_a_total", result.model_a_total)
    record("model_b_total", result.model_b_total)

    assert result.outcome == MatchOutcome.MODEL_B_WINS
    assert result.model_b_total > result.model_a_total


def test_match_auto_win_when_red_missing_challenge(tmp_path, record):
    # A has 2 challenges, B has 0 → B (as Red) can't challenge A, so A wins
    # those 2 turns by default. With turns_per_player=2:
    #   A-Red turns: 2 real (B defends)
    #   B-Red turns: 0 real + 2 auto-wins for A (A is the defender)
    orchestrator, store = _make_orchestrator(
        tmp_path,
        turns_per_player=2,
        blue_composite_when_b_defends=0.0,  # B fails to defend A's challenges
        blue_composite_when_a_defends=0.0,
    )
    _populate_store(store, "vendor/model-a", "mock_repo", 2)
    # B has no challenges for mock_repo.

    result = orchestrator.execute(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=1
    )

    auto_turns = [
        t for t in result.turns
        if t.defense_result.score.test_details.get("auto_win")
    ]
    record("total_turns", len(result.turns))
    record("auto_win_turns", len(auto_turns))
    record("model_a_total", result.model_a_total)
    record("model_b_total", result.model_b_total)
    record("outcome", result.outcome.value)

    # 2 real A-Red turns + 2 auto-win turns (B-Red missing) = 4
    assert len(result.turns) == 4
    assert len(auto_turns) == 2
    # All auto-wins credit A (the defender against B's missing challenges).
    assert all(t.blue_model_id == "vendor/model-a" for t in auto_turns)
    assert result.outcome == MatchOutcome.MODEL_A_WINS


def test_match_multi_repo_aggregates_turns(tmp_path, record):
    orchestrator, store = _make_orchestrator(tmp_path, turns_per_player=1)
    for repo in ("repo1", "repo2", "repo3"):
        _populate_store(store, "vendor/model-a", repo, 1)
        _populate_store(store, "vendor/model-b", repo, 1)

    repos = [
        RepoConfig(
            name=repo, url=f"local://{repo}", commit="abc",
            language="python", test_command="pytest -q", docker_image="swe-duel-mock",
        )
        for repo in ("repo1", "repo2", "repo3")
    ]
    result = orchestrator.execute(
        "vendor/model-a", "vendor/model-b", repos, seed=1
    )

    record("total_turns", len(result.turns))
    record("repo_name", result.repo_name)
    # 1 turn/player/repo × 2 sides × 3 repos = 6 turns, one MatchResult.
    assert len(result.turns) == 6
    assert result.repo_name == "repo1,repo2,repo3"


def test_match_interleaving(tmp_path, record):
    orchestrator, store = _make_orchestrator(tmp_path, turns_per_player=3)
    _populate_store(store, "vendor/model-a", "mock_repo", 4)
    _populate_store(store, "vendor/model-b", "mock_repo", 4)

    result = orchestrator.execute(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=11
    )

    red_sequence = [r.red_model_id for r in result.turns]
    expected = [
        "vendor/model-a",
        "vendor/model-b",
        "vendor/model-a",
        "vendor/model-b",
        "vendor/model-a",
        "vendor/model-b",
    ]

    record("red_sequence", red_sequence)
    record("turn_indices", [r.turn_index for r in result.turns])

    assert red_sequence == expected
    assert [r.turn_index for r in result.turns] == list(range(6))


# ── Phased / parallel API ─────────────────────────────────


def test_plan_match_emits_tasks_for_cache_misses(tmp_path, record):
    # Nothing in the defense cache → every selected challenge becomes a task.
    orchestrator, store = _make_orchestrator(tmp_path, turns_per_player=2)
    _populate_store(store, "vendor/model-a", "mock_repo", 3)
    _populate_store(store, "vendor/model-b", "mock_repo", 3)

    plan = orchestrator.plan_match(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=1
    )

    record("pending_tasks", plan.pending_count())
    record("cache_hits", plan.cache_hits())
    # turns_per_player=2 × 2 sides × 1 repo = 4 tasks, all cache misses.
    assert plan.pending_count() == 4
    assert plan.cache_hits() == 0
    sides = sorted(t.side for t in plan.pending_tasks)
    assert sides == ["a_defends_b", "a_defends_b", "b_defends_a", "b_defends_a"]


def test_phased_api_matches_execute(tmp_path, record):
    # Driving plan → run_defense_task → place_result → assemble_match by hand
    # must reproduce what execute() produces sequentially.
    orchestrator, store = _make_orchestrator(
        tmp_path,
        turns_per_player=2,
        blue_composite_when_b_defends=1.0,
        blue_composite_when_a_defends=0.0,
    )
    _populate_store(store, "vendor/model-a", "mock_repo", 3)
    _populate_store(store, "vendor/model-b", "mock_repo", 3)

    plan = orchestrator.plan_match(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=1
    )
    for task in plan.pending_tasks:
        result = orchestrator.run_defense_task(task)
        orchestrator.place_result(plan, task, result)
    match = orchestrator.assemble_match(plan)

    record("model_a_total", match.model_a_total)
    record("model_b_total", match.model_b_total)
    record("total_turns", len(match.turns))
    assert match.model_a_total == pytest.approx(0.0)
    assert match.model_b_total == pytest.approx(4.0)
    assert len(match.turns) == 4


def test_round_pool_parallel_matches_sequential(tmp_path, record):
    # The same set of tasks run through a ThreadPoolExecutor must yield the same
    # MatchResult as running them sequentially (results are routed by task slot,
    # so completion order is irrelevant).
    from concurrent.futures import ThreadPoolExecutor

    orchestrator, store = _make_orchestrator(
        tmp_path,
        turns_per_player=3,
        blue_composite_when_b_defends=1.0,
        blue_composite_when_a_defends=0.0,
    )
    _populate_store(store, "vendor/model-a", "mock_repo", 5)
    _populate_store(store, "vendor/model-b", "mock_repo", 5)

    plan = orchestrator.plan_match(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=1
    )
    tasks = list(plan.pending_tasks)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(orchestrator.run_defense_task, t): t for t in tasks}
        for fut in futs:
            t = futs[fut]
            orchestrator.place_result(plan, t, fut.result())
    match = orchestrator.assemble_match(plan)

    record("parallel_total_turns", len(match.turns))
    record("model_a_total", match.model_a_total)
    record("model_b_total", match.model_b_total)
    assert len(match.turns) == 6
    assert match.model_a_total == pytest.approx(0.0)
    assert match.model_b_total == pytest.approx(6.0)


def test_failed_subturn_dropped_not_crash(tmp_path, record):
    # If a defense task fails, its slot stays None; assemble_match drops that
    # turn (keeping challenge/defense alignment) instead of crashing the match.
    orchestrator, store = _make_orchestrator(
        tmp_path,
        turns_per_player=2,
        blue_composite_when_b_defends=1.0,
        blue_composite_when_a_defends=1.0,
    )
    _populate_store(store, "vendor/model-a", "mock_repo", 3)
    _populate_store(store, "vendor/model-b", "mock_repo", 3)

    plan = orchestrator.plan_match(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=1
    )
    # Run all but one task; leave one b_defends_a slot None (simulating a drop).
    dropped = next(t for t in plan.pending_tasks if t.side == "b_defends_a")
    for task in plan.pending_tasks:
        if task is dropped:
            continue
        orchestrator.place_result(plan, task, orchestrator.run_defense_task(task))
    match = orchestrator.assemble_match(plan)

    record("total_turns", len(match.turns))
    # 4 total tasks, 1 dropped → 3 scored turns.
    assert len(match.turns) == 3


def test_cache_hit_skips_task(tmp_path, record):
    # A pre-logged defense for one challenge must be reused (cache hit) and NOT
    # produce a task; the rest remain pending.
    orchestrator, store = _make_orchestrator(tmp_path, turns_per_player=2)
    a_records = _populate_store(store, "vendor/model-a", "mock_repo", 3)
    _populate_store(store, "vendor/model-b", "mock_repo", 3)

    # Pre-log a defense: B defending A's first challenge.
    cached = _defense(a_records[0].challenge_id, "vendor/model-b", 1.0)
    orchestrator.artifact_logger.log_defense(cached)

    plan = orchestrator.plan_match(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=1
    )

    record("pending_tasks", plan.pending_count())
    record("cache_hits", plan.cache_hits())
    # 4 slots total, 1 cached → 3 pending tasks, 1 hit.
    assert plan.cache_hits() == 1
    assert plan.pending_count() == 3
    assert not any(
        t.side == "b_defends_a" and t.challenge.challenge_id == a_records[0].challenge_id
        for t in plan.pending_tasks
    )


def test_run_defense_task_survives_logging_failure_loudly(
    tmp_path, record, monkeypatch, capsys
):
    # A defense whose artifact persistence raises must still return its
    # result (the turn keeps its score) — but never silently: a lost record
    # holes the defense cache (future runs re-run the (challenge, blue)
    # pair at real cost, contributions can never ship it), so the failure
    # must warn loudly with the ids an operator needs.
    orchestrator, store = _make_orchestrator(tmp_path, turns_per_player=2)
    _populate_store(store, "vendor/model-a", "mock_repo", 3)
    _populate_store(store, "vendor/model-b", "mock_repo", 3)
    plan = orchestrator.plan_match(
        "vendor/model-a", "vendor/model-b", _repo_config(), seed=1
    )
    task = plan.pending_tasks[0]

    def _boom(result, *args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(orchestrator.artifact_logger, "log_defense", _boom)
    result = orchestrator.run_defense_task(task)

    record("result_returned", result.defense_id)
    assert result.challenge_id == task.challenge.challenge_id
    assert result.defense_id
    err = capsys.readouterr().err
    assert "persisting defense" in err
    assert result.defense_id in err
    assert "RuntimeError: disk on fire" in err


# ── Participant-identity plumbing (effort / provider) ─────────────────────


def test_match_with_effort_provider_identity(tmp_path, record):
    """A match between 4-tuple composite ids draws challenges from the
    identity-scoped pools and stamps every turn with the full identity."""
    from swe_duel.models import composite_id

    orchestrator, store = _make_orchestrator(tmp_path, turns_per_player=1)
    _populate_store(store, "vendor/model-a", "mock_repo", 2, effort="high", provider="cloudflare")
    _populate_store(store, "vendor/model-b", "mock_repo", 2, effort="low", provider="")

    a_cid = composite_id("vendor/model-a", "mini-swe-agent", "high", "cloudflare")
    b_cid = composite_id("vendor/model-b", "mini-swe-agent", "low", "")
    plan = orchestrator.plan_match(a_cid, b_cid, _repo_config(), seed=1)

    record("pending_tasks", plan.pending_count())
    assert plan.pending_count() == 2  # 1 challenge per side, all cache misses
    # Red challenges come from the identity-scoped pools.
    assert [c.challenge_id for c in plan.a_challenges[0]] == [
        *store.list_ids_in_index_order(
            "vendor/model-a", "mock_repo", "mini-swe-agent", "high", "cloudflare"
        )[:1]
    ]
    # Blue tasks carry the defender's identity.
    for task in plan.pending_tasks:
        if task.side == "b_defends_a":
            assert (task.blue_model_id, task.blue_harness_id) == ("vendor/model-b", "mini-swe-agent")
            assert task.blue_reasoning_effort == "low"
            assert task.blue_provider == ""
        else:
            assert (task.blue_model_id, task.blue_harness_id) == ("vendor/model-a", "mini-swe-agent")
            assert task.blue_reasoning_effort == "high"
            assert task.blue_provider == "cloudflare"

    for task in plan.pending_tasks:
        orchestrator.place_result(plan, task, orchestrator.run_defense_task(task))
    match = orchestrator.assemble_match(plan)
    record("turn_identities", [
        {
            "red": t.red_model_id, "red_effort": t.red_reasoning_effort,
            "red_provider": t.red_provider, "blue_effort": t.blue_reasoning_effort,
            "blue_provider": t.blue_provider,
        }
        for t in match.turns
    ])
    for turn in match.turns:
        # TurnResult red/blue ids are the competitor COMPOSITE ids.
        if turn.red_model_id == a_cid:
            assert turn.red_reasoning_effort == "high"
            assert turn.red_provider == "cloudflare"
            assert turn.blue_reasoning_effort == "low"
            assert turn.blue_provider == ""
        else:
            assert turn.red_model_id == b_cid
            assert turn.red_reasoning_effort == "low"
            assert turn.red_provider == ""
            assert turn.blue_reasoning_effort == "high"
            assert turn.blue_provider == "cloudflare"


def test_defense_cache_key_separates_identity(tmp_path, record):
    """A cached defense by the default identity must NOT be reused for the
    same model at a different effort/provider (and vice versa)."""
    from swe_duel.models import composite_id

    orchestrator, store = _make_orchestrator(tmp_path, turns_per_player=1)
    a_records = _populate_store(store, "vendor/model-a", "mock_repo", 1)
    _populate_store(store, "vendor/model-b", "mock_repo", 1)

    # Cache a defense by B under the DEFAULT identity.
    cached = _defense(a_records[0].challenge_id, "vendor/model-b", 1.0)
    orchestrator.artifact_logger.log_defense(cached)

    # Same-identity match → cache hit.
    a_default = composite_id("vendor/model-a", "mini-swe-agent")
    b_default = composite_id("vendor/model-b", "mini-swe-agent")
    plan = orchestrator.plan_match(a_default, b_default, _repo_config(), seed=1)
    record("default_hits", plan.cache_hits())
    assert plan.cache_hits() == 1

    # B at effort=high is a DIFFERENT competitor → cache miss.
    b_high = composite_id("vendor/model-b", "mini-swe-agent", "high", "")
    plan2 = orchestrator.plan_match(a_default, b_high, _repo_config(), seed=1)
    record("identity_shift_hits", plan2.cache_hits())
    assert plan2.cache_hits() == 0
    # One cache-miss task for B defending A's challenge; B's own effort=high
    # pool is empty, so A's side is an auto-win (no task, one extra turn).
    assert plan2.pending_count() == 1
    match2 = orchestrator.assemble_match(plan2)
    auto_wins = [
        tr for tr in match2.turns
        if tr.defense_result.score.test_details.get("auto_win")
    ]
    assert len(auto_wins) == 1
    # B (effort=high) could not generate → A wins the turn by default.
    assert auto_wins[0].blue_model_id == a_default
    assert auto_wins[0].red_model_id == b_high
    assert auto_wins[0].red_reasoning_effort == "high"
