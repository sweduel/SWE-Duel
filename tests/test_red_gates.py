"""Phase 7: Red validation gate tests.

All tests use a pre-crafted ``RedChallenge`` fixture and a mocked
``TestRunner``/``DockerExecutor`` so no Docker or LLM is touched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from swe_duel.config import ArenaConfig, RedGatesConfig, RepoConfig
from swe_duel.models import (
    AgentTrajectory,
    BugType,
    ExecutionResult,
    GateStatus,
    RedChallenge,
    TestExecutionResult,
)
from swe_duel.sandbox.languages import get_adapter
from swe_duel.validation.red_gates import RedGateValidator


# ── Fixtures ───────────────────────────────────────────────


_VALID_DIFF = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,1 +1,13 @@
 x = 1
+def feature():
+    return 42
+def helper_a():
+    return 1
+def helper_b():
+    return 2
+def helper_c():
+    return 3
+def helper_d():
+    return 4
+def helper_e():
+    return 5
+def helper_f():
+    return 6
"""


_VALID_FEATURE_TESTS = """\
def test_feature_one():
    assert 1 == 1
    assert 2 == 2


def test_feature_two():
    assert 3 == 3
"""


_VALID_BUG_TESTS = """\
def test_bug_reveal():
    assert 1 == 2
"""


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


def _repo_config() -> RepoConfig:
    return RepoConfig(
        name="mock",
        url="local://mock",
        commit="HEAD",
        language="python",
        test_command="python -m pytest -x -q",
        docker_image="swe-duel-mock",
    )


def _arena_config() -> ArenaConfig:
    return ArenaConfig(
        red_gates=RedGatesConfig(
            min_diff_lines=5,
            min_test_assertions=2,
            min_test_functions=1,
        )
    )


def _valid_challenge(
    pr_diff: str = _VALID_DIFF,
    modified_file_contents: dict[str, str] | None = None,
    feature_test_code: str = _VALID_FEATURE_TESTS,
    bug_test_code: str | None = _VALID_BUG_TESTS,
) -> RedChallenge:
    return RedChallenge(
        target_files=["src/foo.py"],
        exploration_summary="summary",
        feature_spec="spec",
        feature_rationale="rationale",
        pr_diff=pr_diff,
        modified_file_contents=modified_file_contents
        if modified_file_contents is not None
        else {"src/foo.py": "x = 1\ndef feature():\n    return 42\n"},
        original_file_contents={"src/foo.py": "x = 1\n"},
        feature_test_code=feature_test_code,
        bug_type=BugType.LOGIC_ERROR,
        bug_description="desc",
        bug_location="src/foo.py:feature",
        bug_test_code=bug_test_code,
        agent_trajectory=_trajectory(),
    )


def _mock_test_result(
    passed: bool,
    total: int = 3,
    failed_count: int = 0,
) -> TestExecutionResult:
    passed_count = total - failed_count if passed else max(0, total - failed_count)
    return TestExecutionResult(
        passed=passed,
        total=total,
        passed_count=passed_count,
        failed_count=failed_count if not passed else 0,
        error_count=0,
        failure_messages=[],
        stdout="",
        stderr="",
        duration_ms=1,
    )


def _make_validator(
    existing_result: TestExecutionResult | None = None,
    feature_result: TestExecutionResult | None = None,
    bug_result: TestExecutionResult | None = None,
    lint_rc: int = 0,
) -> tuple[RedGateValidator, MagicMock]:
    runner = MagicMock()
    runner.run_existing_tests.return_value = existing_result or _mock_test_result(True)
    runner.run_injected_tests.side_effect = [
        feature_result or _mock_test_result(True),
        bug_result or _mock_test_result(False, total=1, failed_count=1),
    ]
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        return_code=lint_rc,
        stdout="",
        stderr="",
        timed_out=False,
        duration_ms=1,
    )
    runner.executor = executor
    runner.adapter = get_adapter("python")
    return RedGateValidator(test_runner=runner, config=_arena_config()), runner


# ── Individual gate tests ─────────────────────────────────


def test_gate_diff_valid_pass(record):
    validator, _ = _make_validator()
    result = validator._gate_diff_valid(_valid_challenge())
    record("gate", result)
    assert result.status == GateStatus.PASSED


def test_gate_diff_valid_empty_diff(record):
    validator, _ = _make_validator()
    result = validator._gate_diff_valid(_valid_challenge(pr_diff=""))
    record("gate", result)
    assert result.status == GateStatus.FAILED


def test_gate_existing_tests_pass(record):
    validator, runner = _make_validator(existing_result=_mock_test_result(True))
    result = validator._gate_existing_tests(_repo_config(), _valid_challenge())
    record("gate", result)
    assert result.status == GateStatus.PASSED
    runner.run_existing_tests.assert_called_once()


def test_gate_existing_tests_fail(record):
    validator, _ = _make_validator(
        existing_result=_mock_test_result(False, total=3, failed_count=1)
    )
    result = validator._gate_existing_tests(_repo_config(), _valid_challenge())
    record("gate", result)
    assert result.status == GateStatus.FAILED


def test_gate_feature_tests_pass(record):
    validator, _ = _make_validator(feature_result=_mock_test_result(True))
    result = validator._gate_feature_tests(_valid_challenge())
    record("gate", result)
    assert result.status == GateStatus.PASSED


def test_gate_feature_tests_fail(record):
    validator, _ = _make_validator(
        feature_result=_mock_test_result(False, total=2, failed_count=1)
    )
    result = validator._gate_feature_tests(_valid_challenge())
    record("gate", result)
    assert result.status == GateStatus.FAILED


def test_gate_bug_tests_pass_is_fail(record):
    """All bug tests pass ⇒ no bug present ⇒ gate fails."""
    validator, runner = _make_validator()
    runner.run_injected_tests.side_effect = [_mock_test_result(True, total=2)]
    result = validator._gate_bug_tests(_valid_challenge())
    record("gate", result)
    assert result.status == GateStatus.FAILED


def test_gate_bug_tests_fail_is_pass(record):
    """Bug tests fail ⇒ bug detected ⇒ gate passes."""
    validator, runner = _make_validator()
    runner.run_injected_tests.side_effect = [
        _mock_test_result(False, total=2, failed_count=1)
    ]
    result = validator._gate_bug_tests(_valid_challenge())
    record("gate", result)
    assert result.status == GateStatus.PASSED


def test_gate_complexity_pass(record):
    validator, _ = _make_validator()
    result = validator._gate_complexity(_valid_challenge())
    record("gate", result)
    assert result.status == GateStatus.PASSED


def test_gate_complexity_fail_too_small(record):
    tiny_diff = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,1 +1,2 @@
 x = 1
+y = 2
"""
    validator, _ = _make_validator()
    result = validator._gate_complexity(_valid_challenge(pr_diff=tiny_diff))
    record("gate", result)
    assert result.status == GateStatus.FAILED
    assert "min_diff_lines" in result.message


def test_validate_feature_only_all_pass(record):
    validator, runner = _make_validator()
    result = validator.validate_feature_only(
        _repo_config(), _valid_challenge(pr_diff=_VALID_DIFF)
    )
    record("passed", result.passed)
    record("gate_results", [(g.gate_name, g.status) for g in result.gate_results])
    assert result.passed
    gate_names = [g.gate_name for g in result.gate_results]
    assert gate_names == [
        "gate_existing_tests",
        "gate_feature_tests",
        "gate_complexity",
    ]


def test_validate_feature_only_complexity_fail(record):
    tiny_diff = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,1 +1,2 @@
 x = 1
+y = 2
"""
    validator, runner = _make_validator()
    result = validator.validate_feature_only(
        _repo_config(), _valid_challenge(pr_diff=tiny_diff)
    )
    record("passed", result.passed)
    record("gate_results", [(g.gate_name, g.status) for g in result.gate_results])
    assert not result.passed
    complexity_gate = next(
        (g for g in result.gate_results if g.gate_name == "gate_complexity"), None
    )
    assert complexity_gate is not None
    assert complexity_gate.status == GateStatus.FAILED
    # Complexity failure short-circuits before the bug-embedding gate calls.
    assert runner.run_injected_tests.call_count == 1
    assert runner.run_injected_tests.call_args.kwargs.get("test_filename") == (
        "test_swe_duel_feature.py"
    )


# ── End-to-end validate() tests ───────────────────────────


def test_validate_all_gates_pass(record):
    validator, _ = _make_validator()
    result = validator.validate(_repo_config(), _valid_challenge())
    record("passed", result.passed)
    record("gate_results", [(g.gate_name, g.status) for g in result.gate_results])
    assert result.passed
    assert all(g.status == GateStatus.PASSED for g in result.gate_results)
    assert len(result.gate_results) == 6


def test_validate_short_circuit_on_diff(record):
    validator, runner = _make_validator()
    result = validator.validate(_repo_config(), _valid_challenge(pr_diff=""))
    record("passed", result.passed)
    record("gate_results", [(g.gate_name, g.status) for g in result.gate_results])
    assert not result.passed
    assert len(result.gate_results) == 1
    assert result.gate_results[0].gate_name == "gate_diff_valid"
    runner.run_existing_tests.assert_not_called()
    runner.run_injected_tests.assert_not_called()
    runner.executor.execute.assert_not_called()


# ── Fix-based self-review gate ────────────────────────────


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "src" / "swe_duel" / "agents" / "prompts"


def _make_self_review_validator(
    workspace_root: Path,
    *,
    feature_result: TestExecutionResult | None = None,
    bug_result: TestExecutionResult | None = None,
    post_fix_bug_result: TestExecutionResult | None = None,
    lint_rc: int = 0,
    fixed_files: dict[str, str] | None = None,
) -> tuple[RedGateValidator, MagicMock, MagicMock, MagicMock]:
    """Build a validator with the self-review gate wired up (mocked agent + workspaces).

    ``run_injected_tests`` is consumed in order: feature gate, bug gate, then the
    post-fix bug run inside ``_gate_self_review``.
    """
    runner = MagicMock()
    runner.run_existing_tests.return_value = _mock_test_result(True)
    runner.run_injected_tests.side_effect = [
        feature_result or _mock_test_result(True),
        bug_result or _mock_test_result(False, total=1, failed_count=1),
        post_fix_bug_result or _mock_test_result(True, total=1),
    ]
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        return_code=lint_rc, stdout="", stderr="", timed_out=False, duration_ms=1
    )
    runner.executor = executor
    runner.adapter = get_adapter("python")

    workspace = MagicMock()
    workspace.path = workspace_root
    wm = MagicMock()
    wm.create_workspace.return_value = workspace
    wm.get_modified_files.return_value = fixed_files or {
        "src/foo.py": "x = 1\ndef feature():\n    return 42  # fixed\n"
    }
    aw = MagicMock()
    aw.model_config.model_id = "test/mock"
    aw.run.return_value = _trajectory()

    validator = RedGateValidator(
        test_runner=runner,
        config=_arena_config(),
        agent_wrapper=aw,
        workspace_manager=wm,
        prompt_dir=_PROMPT_DIR,
    )
    return validator, runner, wm, aw


def test_self_review_passes_when_bug_fixed(record, tmp_path):
    """All bug tests pass on the reviewer's fixed code ⇒ gate passes, detected=True."""
    validator, runner, wm, aw = _make_self_review_validator(
        tmp_path, post_fix_bug_result=_mock_test_result(True, total=2)
    )
    result = validator.validate(_repo_config(), _valid_challenge())
    sr_gate = next(g for g in result.gate_results if g.gate_name == "gate_self_review")
    record("passed", result.passed)
    record("self_review_gate", sr_gate)
    record("self_review", result.self_review)
    assert result.passed
    assert sr_gate.status == GateStatus.PASSED
    assert result.self_review is not None and result.self_review.detected is True
    # The reviewer agent actually ran, and the post-fix bug run used its files.
    aw.run.assert_called_once()
    wm.get_modified_files.assert_called_once_with(wm.create_workspace.return_value)
    wm.cleanup.assert_called_once()


def test_self_review_fails_when_bug_not_fixed(record, tmp_path):
    """Bug tests still fail on the reviewer's code ⇒ gate fails, challenge rejected."""
    validator, runner, wm, aw = _make_self_review_validator(
        tmp_path, post_fix_bug_result=_mock_test_result(False, total=2, failed_count=1)
    )
    result = validator.validate(_repo_config(), _valid_challenge())
    sr_gate = next(g for g in result.gate_results if g.gate_name == "gate_self_review")
    record("passed", result.passed)
    record("self_review_gate", sr_gate)
    assert not result.passed
    assert sr_gate.status == GateStatus.FAILED
    assert result.self_review is not None and result.self_review.detected is False


def test_self_review_skipped_when_bug_gate_fails(record, tmp_path):
    """If gate_bug_tests fails, the token-expensive self-review gate must NOT run."""
    validator, runner, wm, aw = _make_self_review_validator(
        tmp_path,
        # bug gate sees the bug tests PASS on bugged code ⇒ no bug ⇒ gate fails.
        bug_result=_mock_test_result(True, total=2),
    )
    result = validator.validate(_repo_config(), _valid_challenge())
    gate_names = [g.gate_name for g in result.gate_results]
    record("passed", result.passed)
    record("gate_results", [(g.gate_name, g.status) for g in result.gate_results])
    assert not result.passed
    assert "gate_self_review" not in gate_names
    aw.run.assert_not_called()
    wm.create_workspace.assert_not_called()


def test_self_review_skipped_when_existing_tests_fail(record, tmp_path):
    """A regression in existing tests also skips the self-review gate."""
    validator, runner, wm, aw = _make_self_review_validator(tmp_path)
    runner.run_existing_tests.return_value = _mock_test_result(
        False, total=3, failed_count=1
    )
    result = validator.validate(_repo_config(), _valid_challenge())
    gate_names = [g.gate_name for g in result.gate_results]
    record("gate_results", [(g.gate_name, g.status) for g in result.gate_results])
    assert not result.passed
    assert "gate_self_review" not in gate_names
    aw.run.assert_not_called()

