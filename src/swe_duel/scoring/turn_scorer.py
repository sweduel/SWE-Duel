"""Deterministic, test-execution-based scoring of Blue fixes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from swe_duel.config import ArenaConfig, RepoConfig
from swe_duel.models import (
    BlueFix,
    ChallengeRecord,
    TurnScore,
    TestExecutionResult,
)
from swe_duel.sandbox.test_runner import TestRunner


class TurnScorer:
    """Run 3 test suites against Blue's workspace and compute composite scores."""

    def __init__(self, test_runner: TestRunner, config: ArenaConfig) -> None:
        self.test_runner = test_runner
        # ScoringConfig lambdas are retained on ArenaConfig for YAML back-compat
        # but are no longer used: every suite is a hard multiplicative gate.
        _ = config

    def score(
        self,
        repo_config: RepoConfig,
        challenge_record: ChallengeRecord,
        blue_fix: BlueFix,
        *,
        progress: Callable[[str, str], None] | None = None,
    ) -> TurnScore:
        """Run the three deterministic test suites against Blue's workspace and
        compute the composite scores.

        ``progress(phase, status)`` (optional) is invoked as each test phase
        ("regression"/"feature"/"bugfix") starts ("running") and resolves
        ("success"/"fail"), so a live UI can show per-phase glyphs.
        """

        def _emit(phase: str, status: str) -> None:
            if progress is not None:
                try:
                    progress(phase, status)
                except Exception:
                    pass

        # (1) Empty fix → all zeros
        if not blue_fix.fix_diff and not blue_fix.modified_file_contents:
            return TurnScore(
                s_regression=0.0,
                s_feature=0.0,
                s_bugfix=0.0,
                blue_composite=0.0,
                red_composite=1.0,
                test_details={},
            )

        file_overrides = blue_fix.modified_file_contents
        test_details: dict = {}

        # (2) Regression
        _emit("regression", "running")
        s_regression, regression_result = self._score_regression(repo_config, file_overrides)
        test_details["regression"] = _serialize(regression_result)
        _emit("regression", "success" if s_regression >= 1.0 else "fail")

        # (3) Hard gate
        if s_regression == 0.0:
            return TurnScore(
                s_regression=0.0,
                s_feature=0.0,
                s_bugfix=0.0,
                blue_composite=0.0,
                red_composite=1.0,
                test_details=test_details,
            )

        # (4) Feature
        _emit("feature", "running")
        s_feature, feature_result = self._score_feature(file_overrides, challenge_record)
        test_details["feature"] = _serialize(feature_result)
        _emit("feature", "success" if s_feature >= 1.0 else "fail")

        # (5) Bugfix
        _emit("bugfix", "running")
        s_bugfix, bugfix_result = self._score_bugfix(file_overrides, challenge_record)
        test_details["bugfix"] = _serialize(bugfix_result)
        _emit("bugfix", "success" if s_bugfix >= 1.0 else "fail")

        # (6) Composite.
        # All three suites are HARD multiplicative gates. Blue wins the turn only
        # by fixing the bug while retaining the feature and not regressing;
        # every other outcome is a total Red win (binary scores, no partial credit).
        blue_composite = s_regression * s_feature * s_bugfix
        red_composite = 1.0 - blue_composite

        return TurnScore(
            s_regression=s_regression,
            s_feature=s_feature,
            s_bugfix=s_bugfix,
            blue_composite=blue_composite,
            red_composite=red_composite,
            test_details=test_details,
        )

    # ── Sub-scores ──────────────────────────────────────────

    def _score_regression(
        self, repo_config: RepoConfig, file_overrides: dict[str, str]
    ) -> tuple[float, TestExecutionResult]:
        result = self.test_runner.run_existing_tests(file_overrides=file_overrides)
        return (1.0 if result.passed else 0.0), result

    def _score_feature(
        self, file_overrides: dict[str, str], challenge_record: ChallengeRecord
    ) -> tuple[float, TestExecutionResult]:
        result = self.test_runner.run_injected_tests(
            file_overrides=file_overrides,
            test_code=challenge_record.challenge.feature_test_code,
            test_filename="test_swe_duel_feature.py",
            target_files=challenge_record.challenge.target_files,
        )
        return (1.0 if result.passed else 0.0), result

    def _score_bugfix(
        self, file_overrides: dict[str, str], challenge_record: ChallengeRecord
    ) -> tuple[float, TestExecutionResult]:
        result = self.test_runner.run_injected_tests(
            file_overrides=file_overrides,
            test_code=challenge_record.challenge.bug_test_code or "",
            test_filename="test_swe_duel_bug.py",
            target_files=challenge_record.challenge.target_files,
        )
        return (1.0 if result.passed else 0.0), result


def _serialize(result: TestExecutionResult) -> dict:
    return asdict(result)
