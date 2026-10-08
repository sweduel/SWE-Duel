"""Test-suite invocation and result parsing, delegated per source language.

Command construction and output parsing vary by `RepoConfig.language` and are
owned by a `LanguageAdapter` (see ``swe_duel.sandbox.languages``). This class drives
the Docker executor, applies retries / majority voting, and manages the
per-image baseline-failure cache.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import defaultdict
from pathlib import Path

from swe_duel.config import RepoConfig
from swe_duel.models import ExecutionResult, TestExecutionResult
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.languages import LanguageAdapter, get_adapter


class TestRunner:
    """Run test suites inside Docker and parse output via a language adapter."""

    # Per-(image, test_command) cache of nodeids that fail on pristine code
    # (e.g. plugin interactions, transitive-dep incompatibilities). These are
    # deselected when evaluating agent-modified code so the gate only sees
    # NEW failures caused by the agent. Only populated for languages whose
    # adapter supports per-test deselection (currently Python/pytest).
    _baseline_cache: dict[tuple[str, str], list[str]] = {}
    # Guards _baseline_cache so concurrent defense sub-turns (round-level
    # ThreadPoolExecutor) don't race the check-then-populate. A per-key lock
    # also ensures the (expensive) pristine baseline run happens at most once
    # per (image, test_command) even under contention.
    _baseline_cache_lock = threading.Lock()
    _baseline_key_locks: dict[tuple[str, str], threading.Lock] = defaultdict(
        threading.Lock
    )

    def __init__(
        self,
        executor: DockerExecutor,
        repo_config: RepoConfig,
        retries: int = 1,
    ) -> None:
        self.executor = executor
        self.repo_config = repo_config
        self.retries = retries
        self.adapter: LanguageAdapter = get_adapter(
            repo_config.language, repo_config.name
        )

    def _sanitize_overrides(self, file_overrides: dict[str, str]) -> dict[str, str]:
        """Strip build-system edits before test execution."""
        return self.adapter.filter_file_overrides(file_overrides)

    def run_existing_tests(
        self,
        file_overrides: dict[str, str],
        extra_deselect: list[str] | None = None,
        stream_file: Path | None = None,
    ) -> TestExecutionResult:
        """Run the repo's test suite with source files overridden. Retries on flaky results.

        Pre-existing failures on pristine code are deselected so the gate only
        catches regressions introduced by the agent. Callers can supply
        ``extra_deselect`` to additionally skip nodeids the Red agent recorded
        as pre-existing failures in its workspace metadata. Deselection is only
        applied for languages whose adapter supports it (pytest).
        """
        file_overrides = self._sanitize_overrides(file_overrides)
        if self.adapter.supports_baseline_deselect():
            baseline_failures = self._get_baseline_failures()
            deselects = list(
                dict.fromkeys([*baseline_failures, *(extra_deselect or [])])
            )
        else:
            deselects = []
        command = self.adapter.build_existing_command(
            self.repo_config.test_command, deselects
        )
        return self._run_with_retries(
            file_overrides=file_overrides,
            extra_files=None,
            command=command,
            stream_file=stream_file,
        )

    def _get_baseline_failures(self) -> list[str]:
        key = (self.executor.docker_image, self.repo_config.test_command)
        # Fast path: already cached.
        with TestRunner._baseline_cache_lock:
            if key in TestRunner._baseline_cache:
                return TestRunner._baseline_cache[key]
            key_lock = TestRunner._baseline_key_locks[key]

        # Serialize the expensive pristine run per key, so concurrent callers
        # for the same (image, test_command) compute the baseline once.
        with key_lock:
            with TestRunner._baseline_cache_lock:
                if key in TestRunner._baseline_cache:
                    return TestRunner._baseline_cache[key]

            # Run full suite on pristine code (no overrides, no -x) to enumerate
            # every pre-existing failing nodeid.
            pristine_cmd = self.repo_config.test_command.replace(" -x ", " ").replace(" -x", "")
            exec_result = self.executor.execute(
                file_overrides={},
                command=pristine_cmd,
            )
            parsed = self._parse(exec_result.stdout, exec_result)
            failing_nodeids = [_extract_nodeid(m) for m in parsed.failure_messages]
            failing_nodeids = [n for n in failing_nodeids if n]
            with TestRunner._baseline_cache_lock:
                TestRunner._baseline_cache[key] = failing_nodeids
            return failing_nodeids

    def run_injected_tests(
        self,
        file_overrides: dict[str, str],
        test_code: str,
        test_filename: str,
        target_files: list[str] | None = None,
        stream_file: Path | None = None,
    ) -> TestExecutionResult:
        """Write an injected test and run it via the language adapter.

        `test_filename` is a logical label; the adapter remaps it to a
        language-appropriate on-disk path (e.g. ``_swe-duel/..._test.go`` for Go).
        """
        file_overrides = self._sanitize_overrides(file_overrides)
        extra_files, command = self.adapter.prepare_injected(
            test_filename, test_code, target_files
        )
        return self._run_with_retries(
            file_overrides=file_overrides,
            extra_files=extra_files,
            command=command,
            stream_file=stream_file,
        )

    def run_existing_tests_in_workspace(
        self,
        workspace_path: Path,
        stream_file: Path | None = None,
    ) -> TestExecutionResult:
        """Run the repo's test suite in an existing workspace directory."""
        command = self.adapter.build_existing_command(
            self.repo_config.test_command, []
        )
        results: list[TestExecutionResult] = []
        for _ in range(self.retries):
            exec_result = self.executor.execute_in_workspace(
                workspace_path=workspace_path,
                command=command,
                stream_file=stream_file,
            )
            parsed = self._parse(exec_result.stdout, exec_result)
            results.append(parsed)

        return dataclasses.replace(self._majority_vote(results), command=command)

    def _run_with_retries(
        self,
        file_overrides: dict[str, str],
        extra_files: dict[str, str] | None,
        command: str,
        stream_file: Path | None = None,
    ) -> TestExecutionResult:
        results: list[TestExecutionResult] = []
        for _ in range(self.retries):
            exec_result = self.executor.execute(
                file_overrides=file_overrides,
                command=command,
                extra_files=extra_files,
                stream_file=stream_file,
            )
            parsed = self._parse(exec_result.stdout, exec_result)
            results.append(parsed)

        return dataclasses.replace(self._majority_vote(results), command=command)

    def _majority_vote(self, results: list[TestExecutionResult]) -> TestExecutionResult:
        """Return the majority outcome across retries."""
        if len(results) == 1:
            return results[0]

        pass_count = sum(1 for r in results if r.passed)
        fail_count = len(results) - pass_count

        if pass_count > fail_count:
            representative = next(r for r in results if r.passed)
        else:
            representative = next(r for r in results if not r.passed)

        return representative

    def _parse(
        self, stdout: str, exec_result: ExecutionResult
    ) -> TestExecutionResult:
        return self.adapter.parse(stdout, exec_result)

    # Backwards-compatible alias used by unit tests that parse pytest output
    # directly. Always parses with the Python adapter regardless of repo lang.
    def _parse_pytest_json(
        self, stdout: str, exec_result: ExecutionResult
    ) -> TestExecutionResult:
        return get_adapter("python").parse(stdout, exec_result)


def _extract_nodeid(failure_message: str) -> str:
    """Extract the nodeid from a failure message.

    JSON parser emits ``"{nodeid}: {msg}"``; text parser emits the raw
    ``FAILED <nodeid> - <msg>`` tail. Normalize both.
    """
    msg = failure_message.strip()
    # Text form: "<nodeid> - <msg>" (from re.finditer(r"FAILED (.+)"))
    if " - " in msg:
        candidate = msg.split(" - ", 1)[0].strip()
        if "::" in candidate:
            return candidate
    # JSON form: "<nodeid>: <msg>"
    if ": " in msg:
        candidate = msg.split(": ", 1)[0].strip()
        if "::" in candidate:
            return candidate
    return msg.split()[0] if msg else ""
