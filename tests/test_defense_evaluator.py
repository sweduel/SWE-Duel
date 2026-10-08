"""Phase 11: DefenseEvaluator unit tests (mocked Blue + scorer)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swe_duel.config import ArenaConfig, RepoConfig
from swe_duel.engine.defense_evaluator import DefenseEvaluator, StaleChallengeError
from swe_duel.models import (
    AgentTrajectory,
    BlueFix,
    ChallengeRecord,
    GateResult,
    GateStatus,
    RedChallenge,
    RedValidationResult,
    TurnScore,
    Workspace,
)


# ── Builders ──────────────────────────────────────────────


def _trajectory(cost: float = 0.05) -> AgentTrajectory:
    return AgentTrajectory(
        steps=[],
        total_steps=1,
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost_usd=cost,
        model_id="blue/mock",
        duration_seconds=0.1,
    )


def _red_challenge() -> RedChallenge:
    return RedChallenge(
        target_files=["src/foo.py"],
        exploration_summary="e",
        feature_spec="s",
        feature_rationale="r",
        pr_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n x\n+y\n",
        modified_file_contents={"src/foo.py": "content"},
        original_file_contents={"src/foo.py": "orig"},
        feature_test_code="def test_f():\n    assert True\n",
        bug_type="loop bound skips the final element",
        bug_description="d",
        bug_location="l",
        bug_test_code="def test_b():\n    assert True\n",
        agent_trajectory=_trajectory(),
    )


def _challenge_record(commit: str = "abc123") -> ChallengeRecord:
    return ChallengeRecord(
        challenge_id=str(uuid.uuid4()),
        red_model_id="red/model",
        repo_name="mock_repo",
        repo_commit_sha=commit,
        target_files=["src/foo.py"],
        challenge=_red_challenge(),
        validation=RedValidationResult(
            passed=True,
            gate_results=[
                GateResult(
                    gate_name="g",
                    status=GateStatus.PASSED,
                    message="ok",
                )
            ],
            attempt_number=1,
        ),
        generated_at=datetime.now(timezone.utc),
        generation_cost_usd=0.01,
        generation_retries=0,
    )


def _repo_config(commit: str = "abc123") -> RepoConfig:
    return RepoConfig(
        name="mock_repo",
        url="local://mock_repo",
        commit=commit,
        language="python",
        test_command="pytest -q",
        docker_image="swe-duel-mock",
    )


def _blue_fix() -> BlueFix:
    return BlueFix(
        review_findings=[],
        fix_explanation="fixed",
        fix_diff="--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y\n",
        modified_file_contents={"src/foo.py": "fixed"},
        agent_trajectory=_trajectory(cost=0.07),
    )


def _round_score(blue: float = 1.0) -> TurnScore:
    return TurnScore(
        s_regression=1.0,
        s_feature=1.0,
        s_bugfix=blue,
        blue_composite=blue,
        red_composite=1.0 - blue,
        test_details={},
    )


def _make_evaluator(blue_fix: BlueFix, score: TurnScore):
    blue_agent = MagicMock()
    workspace = Workspace(
        workspace_id="ws-1",
        repo_name="mock_repo",
        path=Path("/tmp/does-not-matter"),
        reference_path=Path("/tmp/does-not-matter-ref"),
    )
    blue_agent.review_and_fix.return_value = (blue_fix, workspace)
    blue_agent.workspace_manager = MagicMock()
    blue_agent.workspace_manager.cleanup = MagicMock()
    blue_agent.agent_wrapper = MagicMock()
    blue_agent.agent_wrapper.model_config = MagicMock()
    blue_agent.agent_wrapper.model_config.model_id = "blue/model"

    scorer = MagicMock()
    scorer.score.return_value = score

    evaluator = DefenseEvaluator(
        blue_agent=blue_agent, scorer=scorer, config=ArenaConfig()
    )
    return evaluator, blue_agent, scorer


# ── Tests ─────────────────────────────────────────────────


def test_evaluate_mock_blue(tmp_path, record):
    fix = _blue_fix()
    score = _round_score(blue=0.8)
    evaluator, blue_agent, scorer = _make_evaluator(fix, score)

    challenge = _challenge_record(commit="abc123")
    repo_cfg = _repo_config(commit="abc123")

    result = evaluator.evaluate(repo_cfg, challenge)

    record("defense_id_type", type(result.defense_id).__name__)
    record("challenge_id", result.challenge_id)
    record("blue_model_id", result.blue_model_id)
    record("cost_usd", result.cost_usd)
    record(
        "score",
        {
            "blue_composite": result.score.blue_composite,
            "red_composite": result.score.red_composite,
        },
    )

    assert result.challenge_id == challenge.challenge_id
    assert result.blue_model_id == "blue/model"
    assert result.blue_fix is fix
    assert result.score is score
    assert result.cost_usd == pytest.approx(0.07)
    assert blue_agent.review_and_fix.call_count == 1
    assert scorer.score.call_count == 1
    assert blue_agent.workspace_manager.cleanup.call_count == 1


def test_evaluate_stale_challenge(record):
    fix = _blue_fix()
    score = _round_score()
    evaluator, blue_agent, scorer = _make_evaluator(fix, score)

    challenge = _challenge_record(commit="OLD_SHA")
    repo_cfg = _repo_config(commit="NEW_SHA")

    with pytest.raises(StaleChallengeError) as excinfo:
        evaluator.evaluate(repo_cfg, challenge)

    record("error_message", str(excinfo.value))
    assert "OLD_SHA" in str(excinfo.value)
    assert blue_agent.review_and_fix.call_count == 0
    assert scorer.score.call_count == 0


def test_evaluate_batch(record):
    fix = _blue_fix()
    score = _round_score(blue=0.5)
    evaluator, blue_agent, scorer = _make_evaluator(fix, score)

    challenges = [_challenge_record(commit="abc123") for _ in range(3)]
    repo_cfg = _repo_config(commit="abc123")

    results = evaluator.evaluate_batch(repo_cfg, challenges)

    record("n_results", len(results))
    record("challenge_ids", [r.challenge_id for r in results])

    assert len(results) == 3
    assert [r.challenge_id for r in results] == [c.challenge_id for c in challenges]
    assert blue_agent.review_and_fix.call_count == 3
    assert scorer.score.call_count == 3


def test_evaluate_warns_when_tui_blue_log_missing(tmp_path, record, capsys):
    # In TUI mode (console echo absorbed) _swe-duel/blue.log is the ONLY
    # surviving record of the Blue agent's reasoning — a missing one must
    # warn, not vanish silently (console runs stream the reasoning to the
    # operator instead, so they stay quiet).
    evaluator, blue_agent, scorer = _make_evaluator(_blue_fix(), _round_score())
    repo_cfg = _repo_config(commit="abc123")
    challenge = _challenge_record(commit="abc123")

    result = evaluator.evaluate(
        repo_cfg, challenge, console_echo=False, log_dir=tmp_path
    )

    record("defense_id", result.defense_id)
    err = capsys.readouterr().err
    assert "no _swe-duel/blue.log" in err
    assert result.defense_id in err


def test_evaluate_quiet_when_console_blue_log_missing(tmp_path, capsys):
    # Console mode (echo on): the operator saw the reasoning live; a missing
    # workspace log is expected there and never warns.
    evaluator, _blue_agent, _scorer = _make_evaluator(_blue_fix(), _round_score())

    evaluator.evaluate(
        _repo_config(commit="abc123"),
        _challenge_record(commit="abc123"),
        console_echo=True,
        log_dir=tmp_path,
    )

    assert "blue.log" not in capsys.readouterr().err


def test_evaluate_warns_when_workspace_cleanup_fails(tmp_path, record, capsys):
    # A cleanup failure leaks a multi-MB workspace per defense — the operator
    # must hear about it before the drive fills.
    evaluator, blue_agent, scorer = _make_evaluator(_blue_fix(), _round_score())
    blue_agent.workspace_manager.cleanup.side_effect = RuntimeError(
        "root-owned residue"
    )

    result = evaluator.evaluate(
        _repo_config(commit="abc123"),
        _challenge_record(commit="abc123"),
        console_echo=False,
        log_dir=tmp_path,
    )

    record("cleanup_warning", True)
    err = capsys.readouterr().err
    assert "cleaning up workspace" in err
    assert "RuntimeError" in err
    assert result.defense_id  # the defense itself still succeeded
