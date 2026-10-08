"""Red challenge validation gates."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from swe_duel.agents.agent_wrapper import CONTAINER_WORKSPACE
from swe_duel.config import ArenaConfig, RepoConfig
from swe_duel.models import (
    GateResult,
    GateStatus,
    RedChallenge,
    RedSelfReview,
    RedValidationResult,
    ReviewFinding,
)
from swe_duel.sandbox.diff_utils import compute_diff_stats, validate_diff_format
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.validation import complexity


class RedGateValidator:
    """Run the Red validation gates sequentially, short-circuiting on gate 0.

    If `agent_wrapper`, `workspace_manager`, and `prompt_dir` are provided, an
    extra `gate_self_review` runs — but ONLY after every cheaper gate
    (existing/feature/bug-tests/complexity/lint) has passed, since it spins up a
    full emulated-Blue agent and is by far the most token-expensive gate. The
    same Red model is asked to emulate Blue on its own PR (without bug
    knowledge) and to actually FIX the suspected defect, just like the real Blue
    agent it emulates. The gate then runs the hidden bug tests against the
    reviewer's fixed code: it PASSES iff all bug tests now pass (the bug was
    genuinely found and removed), proving the challenge is solvable and fair.
    This rejects challenges whose "bugs" a same-skill reviewer cannot fix — e.g.
    no-op PRs with no real defect (which would unfairly score an accepting
    same-skill Blue as a loser) or bugs that are not realistically fixable.
    """

    def __init__(
        self,
        test_runner: TestRunner,
        config: ArenaConfig,
        agent_wrapper=None,
        workspace_manager=None,
        prompt_dir: Path | None = None,
        self_review_wall_seconds: float | None = None,
        self_review_steps: int = 50,
        log_dir: Path | None = None,
    ) -> None:
        self.test_runner = test_runner
        self.config = config
        self.agent_wrapper = agent_wrapper
        self.workspace_manager = workspace_manager
        self.prompt_dir = prompt_dir
        self.self_review_wall_seconds = self_review_wall_seconds
        self.self_review_steps = self_review_steps
        self.log_dir = log_dir
        self._current_phase = "default"
        self._self_review_template = None
        if prompt_dir is not None:
            env = Environment(
                loader=FileSystemLoader(str(prompt_dir)),
                undefined=StrictUndefined,
                keep_trailing_newline=True,
            )
            self._self_review_template = env.get_template("red_self_review_task.md")

    def validate(
        self,
        repo_config: RepoConfig,
        challenge: RedChallenge,
        attempt_number: int = 1,
        progress: Callable[[str, int, int], None] | None = None,
        console_echo: bool = True,
        phase: str = "full",
    ) -> RedValidationResult:
        old_phase = self._current_phase
        self._current_phase = phase
        try:
            return self._validate(
                repo_config=repo_config,
                challenge=challenge,
                attempt_number=attempt_number,
                progress=progress,
                console_echo=console_echo,
            )
        finally:
            self._current_phase = old_phase

    def _validate(
        self,
        repo_config: RepoConfig,
        challenge: RedChallenge,
        attempt_number: int = 1,
        progress: Callable[[str, int, int], None] | None = None,
        console_echo: bool = True,
    ) -> RedValidationResult:
        results: list[GateResult] = []
        self_review: RedSelfReview | None = None

        if self.config.red_gates.enable_diff_gate:
            gate0 = self._tracked(
                "gate_diff_valid",
                lambda: self._gate_diff_valid(challenge),
                progress,
            )
            results.append(gate0)
            if gate0.status != GateStatus.PASSED:
                return RedValidationResult(
                    passed=False,
                    gate_results=results,
                    attempt_number=attempt_number,
                    self_review=None,
                )

        results.append(
            self._tracked(
                "gate_existing_tests",
                lambda: self._gate_existing_tests(repo_config, challenge),
                progress,
            )
        )
        results.append(
            self._tracked(
                "gate_feature_tests",
                lambda: self._gate_feature_tests(challenge),
                progress,
            )
        )
        results.append(
            self._tracked(
                "gate_bug_tests",
                lambda: self._gate_bug_tests(challenge),
                progress,
            )
        )

        results.append(
            self._tracked(
                "gate_complexity",
                lambda: self._gate_complexity(challenge),
                progress,
            )
        )
        if self.config.red_gates.enable_lint_gate:
            results.append(
                self._tracked(
                    "gate_lint",
                    lambda: self._gate_lint(repo_config, challenge),
                    progress,
                )
            )

        # The self-review gate spins up an entire emulated-Blue agent (the most
        # token-expensive gate by far), so only run it once every cheaper gate
        # has passed. In particular, if gate_bug_tests failed the bug is not even
        # detectable — there is nothing for the reviewer to fix — so skipping
        # here lets the generator move on to the next attempt sooner and at no
        # token cost.
        if self._self_review_enabled() and all(
            g.status == GateStatus.PASSED for g in results
        ):
            gate, self_review = self._gate_self_review(
                repo_config, challenge,
                progress=progress, console_echo=console_echo,
            )
            results.append(gate)

        passed = all(g.status == GateStatus.PASSED for g in results)
        return RedValidationResult(
            passed=passed,
            gate_results=results,
            attempt_number=attempt_number,
            self_review=self_review,
        )

    def _self_review_enabled(self) -> bool:
        return (
            getattr(self.config.red_gates, "enable_self_review_gate", True)
            and self.agent_wrapper is not None
            and self.workspace_manager is not None
            and self._self_review_template is not None
        )

    def run_self_review_probe(
        self,
        repo_config: RepoConfig,
        challenge: RedChallenge,
        progress: Callable[[str, int, int], None] | None = None,
        console_echo: bool = True,
    ) -> tuple[GateResult, RedSelfReview | None]:
        """Run ONE blind self-review-and-fix probe — the Phase C gate, standalone.

        This is the exact admission procedure (``_gate_self_review``) exposed as
        a public one-shot: a fresh workspace with only the PR applied, the same
        model behind this validator's harness acting as a blind reviewer (bug
        description / location / tests withheld), the same artifact-recovery
        loop, and the same hidden-bug-test verdict. It exists so the Phase C
        stability study (``scripts/self_review_stability.py``) can draw
        additional stochastic samples of the admission filter without any drift
        from the original gate — the re-runs stay faithful to admission by
        construction because they ARE the admission code path.

        Returns the gate result (PASSED iff all bug tests pass on the
        reviewer's fixed code) and the parsed self-review, mirroring the
        internal gate's return shape.
        """
        return self._gate_self_review(
            repo_config,
            challenge,
            progress=progress,
            console_echo=console_echo,
        )

    def validate_feature_only(
        self,
        repo_config: RepoConfig,
        challenge: RedChallenge,
        attempt_number: int = 1,
        phase: str = "feature-only",
        progress: Callable[[str, int, int], None] | None = None,
    ) -> RedValidationResult:
        """Run only the gates that are meaningful on a feature-only (pre-bug) challenge.

        Used between Phase A (feature generation) and Phase B (bug embedding) to
        fail fast before spending tokens on bug embedding if the feature itself
        is broken, regresses existing tests, or fails the complexity/non-triviality
        threshold.

        ``progress(phase, step, max_steps)`` — when provided — receives a
        gate-status signal with ``max_steps==0``: ``step==0`` hourglass while
        running, ``step==1`` pass / ``step==2`` fail when done. The generation
        TUI keeps finished glyphs until the next agent phase starts.
        """

        old_phase = self._current_phase
        self._current_phase = phase
        try:
            results: list[GateResult] = [
                self._tracked(
                    "gate_existing_tests",
                    lambda: self._gate_existing_tests(repo_config, challenge),
                    progress,
                ),
                self._tracked(
                    "gate_feature_tests",
                    lambda: self._gate_feature_tests(challenge),
                    progress,
                ),
                self._tracked(
                    "gate_complexity",
                    lambda: self._gate_complexity(challenge),
                    progress,
                ),
            ]
            passed = all(g.status == GateStatus.PASSED for g in results)
            return RedValidationResult(
                passed=passed,
                gate_results=results,
                attempt_number=attempt_number,
            )
        finally:
            self._current_phase = old_phase

    def _tracked(
        self,
        gate_name: str,
        fn: Callable[[], GateResult],
        progress: Callable[[str, int, int], None] | None,
    ) -> GateResult:
        """Run one gate with TUI glyph signals (hourglass → tick/cross).

        Matches ``DefenseRoundReporter`` / ``TurnScorer`` phase semantics:
        - ``progress(name, 0, 0)`` while running (⏳);
        - ``progress(name, 1, 0)`` on pass (✓) or ``(name, 2, 0)`` on fail (✗).

        Finished gates stay on the TUI until the next agent phase starts, which
        clears the gate row batch in ``GenerationReporter.update_phase``.
        """
        if progress is not None:
            try:
                progress(gate_name, 0, 0)
            except Exception:
                pass
        try:
            result = fn()
        except Exception:
            if progress is not None:
                try:
                    progress(gate_name, 2, 0)
                except Exception:
                    pass
            raise
        if progress is not None:
            try:
                step = 1 if result.status == GateStatus.PASSED else 2
                progress(gate_name, step, 0)
            except Exception:
                pass
        return result

    def _gate_log_path(self, gate_name: str) -> Path | None:
        """Return a path for live streaming this gate's container output.

        The file lives under a phase subdirectory so that feature-only and
        full-validation runs produce separate, easy-to-read logs.
        """
        if self.log_dir is None:
            return None
        phase = getattr(self, "_current_phase", "default")
        return self.log_dir / phase / f"{gate_name}.log"

    # ── Gate implementations ──────────────────────────────

    def _gate_diff_valid(self, challenge: RedChallenge) -> GateResult:
        start = time.monotonic()
        if not challenge.pr_diff or not challenge.pr_diff.strip():
            return _fail("gate_diff_valid", "pr_diff is empty", start)
        if not validate_diff_format(challenge.pr_diff):
            return _fail("gate_diff_valid", "pr_diff is not a parseable unified diff", start)
        if not challenge.modified_file_contents:
            return _fail("gate_diff_valid", "modified_file_contents is empty", start)

        stats = compute_diff_stats(challenge.pr_diff)
        return _pass(
            "gate_diff_valid",
            f"+{stats.lines_added}/-{stats.lines_removed} across "
            f"{stats.files_changed} file(s), {stats.hunks} hunk(s)",
            start,
        )

    def _gate_existing_tests(
        self, repo_config: RepoConfig, challenge: RedChallenge
    ) -> GateResult:
        start = time.monotonic()
        log_path = self._gate_log_path("gate_existing_tests")
        result = self.test_runner.run_existing_tests(
            file_overrides=challenge.modified_file_contents,
            extra_deselect=list(challenge.pre_existing_failures or []),
            stream_file=log_path,
        )
        if not result.passed:
            return _fail(
                "gate_existing_tests",
                _format_test_failure("existing tests", result),
                start,
                stdout=result.stdout,
                stderr=result.stderr,
                command=result.command,
            )
        return _pass(
            "gate_existing_tests",
            f"all {result.passed_count}/{result.total} existing tests pass",
            start,
            command=result.command,
        )

    def _gate_feature_tests(self, challenge: RedChallenge) -> GateResult:
        start = time.monotonic()
        if not challenge.feature_test_code or not challenge.feature_test_code.strip():
            return _fail("gate_feature_tests", "feature_test_code is empty", start)

        log_path = self._gate_log_path("gate_feature_tests")
        result = self.test_runner.run_injected_tests(
            file_overrides=challenge.modified_file_contents,
            test_code=challenge.feature_test_code,
            test_filename="test_swe_duel_feature.py",
            target_files=challenge.target_files,
            stream_file=log_path,
        )
        if not result.passed:
            return _fail(
                "gate_feature_tests",
                _format_test_failure("feature tests", result),
                start,
                stdout=result.stdout,
                stderr=result.stderr,
                command=result.command,
            )
        return _pass(
            "gate_feature_tests",
            f"all {result.passed_count}/{result.total} feature tests pass",
            start,
            command=result.command,
        )

    def _gate_bug_tests(self, challenge: RedChallenge) -> GateResult:
        start = time.monotonic()
        if not challenge.bug_test_code or not challenge.bug_test_code.strip():
            return _fail("gate_bug_tests", "bug_test_code is empty", start)

        log_path = self._gate_log_path("gate_bug_tests")
        # (1) Bug tests must FAIL on the bugged code (proves the bug is detectable).
        bugged_result = self.test_runner.run_injected_tests(
            file_overrides=challenge.modified_file_contents,
            test_code=challenge.bug_test_code,
            test_filename="test_swe_duel_bug.py",
            target_files=challenge.target_files,
            stream_file=log_path,
        )
        detected = bugged_result.failed_count + bugged_result.error_count
        if detected < 1:
            return _fail(
                "gate_bug_tests",
                "bug tests all passed on bugged code — bug not detectable",
                start,
                stdout=bugged_result.stdout,
                stderr=bugged_result.stderr,
                command=bugged_result.command,
            )

        # (2) Bug tests must PASS on the pre-bug feature-only code (proves the
        # bug tests are fair — they target the bug, not the feature itself).
        feature_only = challenge.feature_only_file_contents or {}
        if feature_only:
            fair_result = self.test_runner.run_injected_tests(
                file_overrides=feature_only,
                test_code=challenge.bug_test_code,
                test_filename="test_swe_duel_bug.py",
                target_files=challenge.target_files,
                stream_file=log_path,
            )
            if not fair_result.passed:
                return _fail(
                    "gate_bug_tests",
                    (
                        "bug tests did not pass on the pre-bug feature code "
                        f"({fair_result.failed_count} failed, "
                        f"{fair_result.error_count} errored of {fair_result.total}) — "
                        "these tests are unfair: they must only trigger on the embedded bug"
                    ),
                    start,
                    stdout=fair_result.stdout,
                    stderr=fair_result.stderr,
                    command=fair_result.command,
                )

        return _pass(
            "gate_bug_tests",
            f"bug detected by {detected} failing test(s) of {bugged_result.total}; "
            "bug tests pass on pre-bug feature code",
            start,
            command=bugged_result.command,
        )

    def _gate_complexity(self, challenge: RedChallenge) -> GateResult:
        start = time.monotonic()
        passed, message = complexity.check_thresholds(
            challenge.pr_diff,
            challenge.feature_test_code,
            self.config,
            adapter=self.test_runner.adapter,
        )
        if not passed:
            return _fail("gate_complexity", message, start)
        return _pass("gate_complexity", message, start)

    def _gate_self_review(
        self,
        repo_config: RepoConfig,
        challenge: RedChallenge,
        progress: Callable[[str, int, int], None] | None = None,
        console_echo: bool = True,
    ) -> tuple[GateResult, RedSelfReview | None]:
        """Run the Red model as an emulated Blue reviewer that FIXES its own PR.

        The reviewer is given the PR diff + feature spec + feature tests only —
        bug_description, bug_location, and bug_test_code are withheld, exactly as
        the real Blue agent sees them. Unlike the previous review-only gate, the
        reviewer now EDITS the source to fix any defect it finds while retaining
        the feature, then we run the hidden bug tests against its modified files.

        The gate PASSES iff all bug tests pass on the reviewer's fixed code (the
        bug was genuinely found and removed) — this proves a same-skill reviewer
        can solve the challenge, so it is fair for Red to enter the tournament
        with it. It FAILS when the bug tests still fail (the reviewer could not
        fix the bug, e.g. because the PR has no real, fixable defect), preventing
        Red from cheating with unsolvable or no-op challenges.
        """
        start = time.monotonic()
        assert self.workspace_manager is not None
        assert self.agent_wrapper is not None
        assert self._self_review_template is not None

        workspace = self.workspace_manager.create_workspace(
            repo_config,
            model_id=self.agent_wrapper.model_config.model_id,
            role="red-self-review",
        )
        try:
            self.workspace_manager.apply_diff_to_workspace(workspace, challenge.pr_diff)
            swe_duel_dir = workspace.path / "_swe-duel"
            swe_duel_dir.mkdir(exist_ok=True)

            task_prompt = self._self_review_template.render(
                repo_name=repo_config.name,
                workspace_path=CONTAINER_WORKSPACE,
                feature_spec=challenge.feature_spec,
                pr_diff=challenge.pr_diff,
                feature_test_code=challenge.feature_test_code,
                lang=self.test_runner.adapter.name,
            )

            def _missing() -> list[str]:
                return [] if (swe_duel_dir / "review.json").exists() else ["_swe-duel/review.json"]

            def _reminder(items: list[str]) -> str:
                bullets = "\n".join(f"  - {p}" for p in items)
                return (
                    "You have not finished yet. Before you submit, fix the "
                    "suspected defect in the source files (retaining the "
                    "feature) and create:\n"
                    f"{bullets}\n\n"
                    "`_swe-duel/review.json` must contain `findings` (list of "
                    "{location, severity, description}) and `fix_explanation` "
                    "(string). Write it, verify with `cat _swe-duel/review.json`, "
                    "then submit with "
                    "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
                )

            trajectory = self.agent_wrapper.run(
                workspace.path,
                task_prompt,
                verbose=True,
                role_label="red-self-review",
                completion_check=_missing,
                reminder_builder=_reminder,
                max_recovery_turns=3,
                max_wall_seconds=self.self_review_wall_seconds,
                max_steps=self.self_review_steps,
                log_file=swe_duel_dir / "red_self_review_agent.log",
                step_callback=(
                    (lambda c, m: progress("self-review", c, m)) if progress else None
                ),
                console_echo=console_echo,
                docker_image=repo_config.docker_image,
                preserve_paths=repo_config.preserve_paths,
                login_shell=repo_config.login_shell,
            )

            findings, fix_explanation = _parse_review_json(swe_duel_dir / "review.json")

            # Assess the fix the same way Blue is scored: run the hidden bug
            # tests against the reviewer's modified source. The bug is genuinely
            # solvable iff every bug test now passes.
            fixed_files = self.workspace_manager.get_modified_files(workspace)
            fixed_files = {
                rel: content
                for rel, content in fixed_files.items()
                if not rel.startswith("_swe-duel/") and not rel.startswith("_swe-duel")
            }
            bug_result = self.test_runner.run_injected_tests(
                file_overrides=fixed_files,
                test_code=challenge.bug_test_code or "",
                test_filename="test_swe_duel_bug.py",
                target_files=challenge.target_files,
            )
            detected = bool(challenge.bug_test_code) and bug_result.passed

            duration_ms = int((time.monotonic() - start) * 1000)
            if detected:
                reason = (
                    "emulated-Blue reviewer (same Red model) fixed the embedded "
                    f"bug: all {bug_result.passed_count}/{bug_result.total} bug "
                    "tests pass on the reviewer's modified code, confirming the "
                    "challenge is solvable and fair"
                )
                self_review = RedSelfReview(
                    detected=True,
                    findings=findings,
                    fix_explanation=fix_explanation,
                    agent_trajectory=trajectory,
                    detection_reason=reason,
                )
                return (
                    GateResult(
                        gate_name="gate_self_review",
                        status=GateStatus.PASSED,
                        message=reason,
                        stdout=bug_result.stdout,
                        stderr=bug_result.stderr,
                        duration_ms=duration_ms,
                        command=bug_result.command,
                    ),
                    self_review,
                )
            reason = (
                "emulated-Blue reviewer (same Red model) did NOT fix the "
                f"embedded bug ({bug_result.failed_count} failed, "
                f"{bug_result.error_count} errored of {bug_result.total} bug "
                "tests still failing on the reviewer's code) — a same-skill "
                "reviewer cannot solve this challenge; rejecting as unfair"
            )
            self_review = RedSelfReview(
                detected=False,
                findings=findings,
                fix_explanation=fix_explanation,
                agent_trajectory=trajectory,
                detection_reason=reason,
            )
            return (
                GateResult(
                    gate_name="gate_self_review",
                    status=GateStatus.FAILED,
                    message=reason,
                    stdout=bug_result.stdout,
                    stderr=bug_result.stderr,
                    duration_ms=duration_ms,
                    command=bug_result.command,
                ),
                self_review,
            )
        finally:
            self.workspace_manager.cleanup(workspace)

    def _gate_lint(
        self, repo_config: RepoConfig, challenge: RedChallenge
    ) -> GateResult:
        start = time.monotonic()
        files = list(challenge.modified_file_contents.keys())
        if not files:
            return _fail("gate_lint", "no files to lint", start)

        command = self.test_runner.adapter.lint_command(files)
        if command is None:
            return _pass(
                "gate_lint",
                f"no lint configured for language "
                f"{self.test_runner.adapter.name!r} — skipped",
                start,
            )
        exec_result = self.test_runner.executor.execute(
            file_overrides=challenge.modified_file_contents,
            command=command,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        if exec_result.return_code != 0:
            return GateResult(
                gate_name="gate_lint",
                status=GateStatus.FAILED,
                message=f"lint/type check failed (rc={exec_result.return_code})",
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                duration_ms=duration_ms,
                command=command,
            )
        return GateResult(
            gate_name="gate_lint",
            status=GateStatus.PASSED,
            message="ruff + mypy clean",
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            duration_ms=duration_ms,
            command=command,
        )


# ── helpers ───────────────────────────────────────────────


def _format_test_failure(label: str, result) -> str:
    """Build a diagnostic message for a failing test run.

    Surfaces the underlying cause when pytest collected zero tests, instead of
    the misleading ``0 failed, 0 errored`` string.
    """
    if result.total == 0:
        reason = (result.failure_messages[0] if result.failure_messages else "no tests collected")
        return f"{label} could not run (total=0): {reason}"
    detail = f"{result.failed_count} failed, {result.error_count} errored of {result.total}"
    if result.failure_messages:
        first = result.failure_messages[0]
        detail = f"{detail} — {first[:200]}"
    return f"{label} failed: {detail}"


def _pass(name: str, message: str, start: float, command: str = "") -> GateResult:
    return GateResult(
        gate_name=name,
        status=GateStatus.PASSED,
        message=message,
        duration_ms=int((time.monotonic() - start) * 1000),
        command=command,
    )


def _parse_review_json(path: Path) -> tuple[list[ReviewFinding], str]:
    """Parse the emulated-Blue reviewer's review.json. Missing/malformed → empty."""
    if not path.exists():
        return [], ""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return [], ""
    if not isinstance(data, dict):
        return [], ""
    findings: list[ReviewFinding] = []
    raw = data.get("findings") or []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            findings.append(
                ReviewFinding(
                    location=str(item.get("location", "")),
                    severity=str(item.get("severity", "info")).lower().strip(),
                    description=str(item.get("description", "")),
                )
            )
    expl = data.get("fix_explanation")
    fix_explanation = expl if isinstance(expl, str) else ""
    return findings, fix_explanation


def _fail(
    name: str,
    message: str,
    start: float,
    stdout: str = "",
    stderr: str = "",
    command: str = "",
) -> GateResult:
    return GateResult(
        gate_name=name,
        status=GateStatus.FAILED,
        message=message,
        stdout=stdout,
        stderr=stderr,
        duration_ms=int((time.monotonic() - start) * 1000),
        command=command,
    )
