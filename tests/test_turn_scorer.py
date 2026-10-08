"""Phase 10: TurnScorer tests.

Unit tests mock TestRunner for deterministic outcomes. The integration test
uses the real mock_repo Docker image and exercises the sample Red and Blue
fixtures.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swe_duel.config import ArenaConfig, RepoConfig, ScoringConfig
from swe_duel.models import (
    AgentTrajectory,
    BlueFix,
    BugType,
    ChallengeRecord,
    GateResult,
    GateStatus,
    RedChallenge,
    RedValidationResult,
    TurnScore,
    TestExecutionResult,
)
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.scoring.turn_scorer import TurnScorer

from conftest import FIXTURES_DIR as FIXTURES

MOCK_IMAGE = "swe-duel-mock:latest"


# ── Builders ───────────────────────────────────────────────


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


def _arena_config(lambda_feature: float = 0.4, lambda_bugfix: float = 0.6) -> ArenaConfig:
    return ArenaConfig(
        scoring=ScoringConfig(
            lambda_feature=lambda_feature, lambda_bugfix=lambda_bugfix
        )
    )


def _ter(passed: bool, total: int = 2, failed: int = 0) -> TestExecutionResult:
    return TestExecutionResult(
        passed=passed,
        total=total,
        passed_count=total - failed if passed else max(0, total - failed),
        failed_count=0 if passed else max(failed, 1),
        error_count=0,
        failure_messages=[],
        stdout="",
        stderr="",
        duration_ms=1,
    )


def _challenge(
    feature_test_code: str = "def test_f(): assert True\n",
    bug_test_code: str = "def test_b(): assert True\n",
    modified_file_contents: dict[str, str] | None = None,
) -> RedChallenge:
    return RedChallenge(
        target_files=["src/foo.py"],
        exploration_summary="",
        feature_spec="",
        feature_rationale="",
        pr_diff="--- a\n+++ b\n@@\n+x=1\n",
        modified_file_contents=modified_file_contents or {"src/foo.py": "x = 1\n"},
        original_file_contents={"src/foo.py": ""},
        feature_test_code=feature_test_code,
        bug_type=BugType.LOGIC_ERROR,
        bug_description="d",
        bug_location="l",
        bug_test_code=bug_test_code,
        agent_trajectory=_trajectory(),
    )


def _challenge_record(challenge: RedChallenge | None = None) -> ChallengeRecord:
    c = challenge or _challenge()
    return ChallengeRecord(
        challenge_id="cid",
        red_model_id="red/m",
        repo_name="mock",
        repo_commit_sha="abc",
        target_files=c.target_files,
        challenge=c,
        validation=RedValidationResult(
            passed=True,
            gate_results=[
                GateResult(
                    gate_name="gate_diff_valid",
                    status=GateStatus.PASSED,
                    message="ok",
                )
            ],
            attempt_number=1,
        ),
        generated_at=datetime(2026, 4, 19),
        generation_cost_usd=0.0,
        generation_retries=0,
    )


def _blue_fix(
    modified: dict[str, str] | None = None,
    diff: str = "--- a\n+++ b\n@@\n+fix\n",
) -> BlueFix:
    return BlueFix(
        review_findings=[],
        fix_explanation="",
        fix_diff=diff,
        modified_file_contents=modified if modified is not None else {"src/foo.py": "x = 2\n"},
        agent_trajectory=_trajectory(),
    )


def _repo_config(image: str = "swe-duel-mock") -> RepoConfig:
    return RepoConfig(
        name="mock",
        url="",
        commit="",
        test_command="python -m pytest tests/ -x -q --tb=short --json-report --json-report-file=report.json",
        docker_image=image,
    )


def _make_scorer(
    existing: TestExecutionResult | None = None,
    feature: TestExecutionResult | None = None,
    bugfix: TestExecutionResult | None = None,
    arena: ArenaConfig | None = None,
) -> tuple[TurnScorer, MagicMock]:
    runner = MagicMock()
    runner.run_existing_tests.return_value = existing or _ter(True)
    runner.run_injected_tests.side_effect = [
        feature or _ter(True),
        bugfix or _ter(True),
    ]
    scorer = TurnScorer(test_runner=runner, config=arena or _arena_config())
    return scorer, runner


def _score_dict(s: TurnScore) -> dict:
    return {
        "s_regression": s.s_regression,
        "s_feature": s.s_feature,
        "s_bugfix": s.s_bugfix,
        "blue_composite": s.blue_composite,
        "red_composite": s.red_composite,
    }


# ── Unit tests ─────────────────────────────────────────────


def test_score_perfect_fix(record):
    scorer, _ = _make_scorer()
    s = scorer.score(_repo_config(), _challenge_record(), _blue_fix())
    record("score", _score_dict(s))
    assert s.s_regression == 1.0
    assert s.s_feature == 1.0
    assert s.s_bugfix == 1.0
    assert s.blue_composite == pytest.approx(1.0)
    assert s.red_composite == pytest.approx(0.0)


def test_score_progress_callback_fires_phases(record):
    # progress(phase, status) must fire running→verdict for each phase that
    # actually runs, in order.
    scorer, _ = _make_scorer(bugfix=_ter(False, total=1, failed=1))
    events: list[tuple[str, str]] = []
    scorer.score(
        _repo_config(), _challenge_record(), _blue_fix(),
        progress=lambda ph, st: events.append((ph, st)),
    )
    record("events", events)
    assert events == [
        ("regression", "running"),
        ("regression", "success"),
        ("feature", "running"),
        ("feature", "success"),
        ("bugfix", "running"),
        ("bugfix", "fail"),
    ]


def test_score_progress_stops_on_regression_gate(record):
    # If regression fails, feature/bugfix never run → only regression events.
    scorer, _ = _make_scorer(existing=_ter(False, total=2, failed=1))
    events: list[tuple[str, str]] = []
    scorer.score(
        _repo_config(), _challenge_record(), _blue_fix(),
        progress=lambda ph, st: events.append((ph, st)),
    )
    record("events", events)
    assert events == [("regression", "running"), ("regression", "fail")]


def test_score_missed_bug(record):
    # Bug not removed → s_bugfix is a hard gate, so Blue scores 0.0 and Red 1.0
    # regardless of feature retention (accept-the-PR-as-is is a total loss).
    scorer, _ = _make_scorer(bugfix=_ter(False, total=1, failed=1))
    s = scorer.score(_repo_config(), _challenge_record(), _blue_fix())
    record("score", _score_dict(s))
    assert s.s_regression == 1.0
    assert s.s_feature == 1.0
    assert s.s_bugfix == 0.0
    assert s.blue_composite == pytest.approx(0.0)
    assert s.red_composite == pytest.approx(1.0)


def test_score_broke_feature(record):
    # Feature retained is a hard gate: bug fixed but feature lost → Blue 0 / Red 1.
    scorer, _ = _make_scorer(feature=_ter(False, total=1, failed=1))
    s = scorer.score(_repo_config(), _challenge_record(), _blue_fix())
    record("score", _score_dict(s))
    assert s.s_regression == 1.0
    assert s.s_feature == 0.0
    assert s.s_bugfix == 1.0
    assert s.blue_composite == pytest.approx(0.0)
    assert s.red_composite == pytest.approx(1.0)


def test_score_regression(record):
    scorer, runner = _make_scorer(existing=_ter(False, total=3, failed=2))
    s = scorer.score(_repo_config(), _challenge_record(), _blue_fix())
    record("score", _score_dict(s))
    assert s.s_regression == 0.0
    assert s.s_feature == 0.0
    assert s.s_bugfix == 0.0
    assert s.blue_composite == 0.0
    assert s.red_composite == 1.0
    runner.run_injected_tests.assert_not_called()


def test_score_empty_fix(record):
    scorer, runner = _make_scorer()
    fix = _blue_fix(modified={}, diff="")
    s = scorer.score(_repo_config(), _challenge_record(), fix)
    record("score", _score_dict(s))
    assert s.s_regression == 0.0
    assert s.s_feature == 0.0
    assert s.s_bugfix == 0.0
    assert s.blue_composite == 0.0
    assert s.red_composite == 1.0
    runner.run_existing_tests.assert_not_called()
    runner.run_injected_tests.assert_not_called()


def test_score_any_failed_suite_is_total_loss(record):
    # Lambdas (if present in ArenaConfig) no longer affect scoring: any failed
    # suite yields blue_composite=0.0 / red_composite=1.0.
    arena = _arena_config(lambda_feature=0.4, lambda_bugfix=0.6)
    scorer, _ = _make_scorer(
        feature=_ter(True), bugfix=_ter(False, total=1, failed=1), arena=arena
    )
    s = scorer.score(_repo_config(), _challenge_record(), _blue_fix())
    record("score_missed_bug", _score_dict(s))
    assert s.blue_composite == pytest.approx(0.0)
    assert s.red_composite == pytest.approx(1.0)

    scorer2, _ = _make_scorer(
        feature=_ter(False, total=1, failed=1), bugfix=_ter(True), arena=arena
    )
    s2 = scorer2.score(_repo_config(), _challenge_record(), _blue_fix())
    record("score_broke_feature", _score_dict(s2))
    assert s2.blue_composite == pytest.approx(0.0)
    assert s2.red_composite == pytest.approx(1.0)


# ── Integration tests ─────────────────────────────────────


def _image_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", name], capture_output=True
    )
    return result.returncode == 0


def _read(p: Path) -> str:
    return p.read_text()


@pytest.fixture()
def real_runner() -> TestRunner:
    if not _image_exists(MOCK_IMAGE):
        pytest.skip(f"Docker image {MOCK_IMAGE} not built — run: swe-duel setup docker")
    executor = DockerExecutor(docker_image=MOCK_IMAGE, timeout_s=60, memory_mb=256)
    return TestRunner(executor=executor, repo_config=_repo_config(MOCK_IMAGE), retries=1)


@pytest.fixture()
def modulo_challenge_record() -> ChallengeRecord:
    red_dir = FIXTURES / "sample_red_workspace"
    feature_tests = _read(red_dir / "_swe-duel" / "feature_tests.py")
    bug_tests = _read(red_dir / "_swe-duel" / "bug_tests.py")
    buggy_basic = _read(red_dir / "src" / "calculator" / "basic.py")

    challenge = RedChallenge(
        target_files=["src/calculator/basic.py"],
        exploration_summary="",
        feature_spec="modulo()",
        feature_rationale="",
        pr_diff="--- a/src/calculator/basic.py\n+++ b/src/calculator/basic.py\n@@\n+def modulo\n",
        modified_file_contents={"src/calculator/basic.py": buggy_basic},
        original_file_contents={},
        feature_test_code=feature_tests,
        bug_type=BugType.LOGIC_ERROR,
        bug_description="hidden",
        bug_location="src/calculator/basic.py:modulo",
        bug_test_code=bug_tests,
        agent_trajectory=_trajectory(),
    )
    return _challenge_record(challenge)


@pytest.mark.integration
def test_score_with_real_tests(real_runner, modulo_challenge_record, record):
    """Blue's sample (correct) fix → s_regression=1, s_feature=1, s_bugfix=1."""
    blue_basic = _read(
        FIXTURES / "sample_blue_workspace" / "src" / "calculator" / "basic.py"
    )
    blue_fix = BlueFix(
        review_findings=[],
        fix_explanation="use Python %",
        fix_diff="--- a/src/calculator/basic.py\n+++ b/src/calculator/basic.py\n@@\n-return a - int(a / b) * b\n+return a % b\n",
        modified_file_contents={"src/calculator/basic.py": blue_basic},
        agent_trajectory=_trajectory(),
    )
    scorer = TurnScorer(test_runner=real_runner, config=_arena_config())
    s = scorer.score(_repo_config(MOCK_IMAGE), modulo_challenge_record, blue_fix)
    record("score", _score_dict(s))
    record("test_details", s.test_details)
    assert s.s_regression == 1.0
    assert s.s_feature == 1.0
    assert s.s_bugfix == 1.0
    assert s.blue_composite == pytest.approx(1.0)


@pytest.mark.integration
def test_score_with_real_tests_buggy(real_runner, modulo_challenge_record, record):
    """Blue left bug in place (submits Red's buggy file) → s_bugfix=0."""
    buggy_basic = _read(
        FIXTURES / "sample_red_workspace" / "src" / "calculator" / "basic.py"
    )
    blue_fix = BlueFix(
        review_findings=[],
        fix_explanation="no-op",
        fix_diff="--- a/src/calculator/basic.py\n+++ b/src/calculator/basic.py\n@@\n unchanged\n",
        modified_file_contents={"src/calculator/basic.py": buggy_basic},
        agent_trajectory=_trajectory(),
    )
    scorer = TurnScorer(test_runner=real_runner, config=_arena_config())
    s = scorer.score(_repo_config(MOCK_IMAGE), modulo_challenge_record, blue_fix)
    record("score", _score_dict(s))
    record("test_details", s.test_details)
    assert s.s_regression == 1.0
    assert s.s_feature == 1.0
    assert s.s_bugfix == 0.0
    assert s.blue_composite == pytest.approx(0.0)
    assert s.red_composite == pytest.approx(1.0)
