"""Node / TypeScript language adapter.

Per-repo test-runner settings (tsx vs mocha, extension, mocha --require) live in
``profiles/<repo>.yaml``.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from swe_duel.models import ExecutionResult, TestExecutionResult
from swe_duel.sandbox.languages.base import LanguageAdapter, infer_collection_error
from swe_duel.sandbox.languages.profile_loader import load_repo_profile


class NodeAdapter(LanguageAdapter):
    """Adapter for Node projects.

    Two test-runner modes are supported, selected via the repo profile:

      - ``node_test`` (default, used by helmet): the built-in ``node:test``
        runner driven through ``tsx`` for TypeScript sources. Injected tests
        live at ``_swe-duel/*.test.ts`` and are run with ``npx tsx --test <file>``.
      - ``mocha`` (used by expressjs): plain JavaScript + mocha. Injected tests
        live at ``_swe-duel/*.test.js`` and are run with
        ``npx mocha --reporter tap <file>``.
    """

    name = "node"

    def __init__(self, repo_name: str | None = None) -> None:
        self.repo_name = (repo_name or "").strip()
        self._profile = load_repo_profile(self.repo_name)
        self._runner = str(self._profile.get("test_runner", "node_test"))

    def _ts_test_path(self, test_filename: str) -> str:
        stem = PurePosixPath(test_filename).stem
        stem = re.sub(r"[^0-9A-Za-z_.-]", "_", stem)
        ext = str(self._profile.get("ext", "test.ts"))
        return f"_swe-duel/{stem}.{ext}"

    @property
    def is_mocha(self) -> bool:
        return self._runner == "mocha"

    def prepare_injected(
        self,
        test_filename: str,
        test_code: str,
        target_files: list[str] | None,
    ) -> tuple[dict[str, str], str]:
        path = self._ts_test_path(test_filename)
        if self.is_mocha:
            require_clause = ""
            require = self._profile.get("require")
            if require:
                require_clause = f" --require {require}"
            command = f"npx mocha{require_clause} --reporter tap {path}"
        else:
            command = f"npx tsx --test {path}"
        return {path: test_code}, command

    def build_existing_command(self, base_command: str, deselects: list[str]) -> str:
        if self.is_mocha:
            # The repo's test_command may use mocha's spec/dot reporters (nice
            # for humans), but the adapter parses TAP. Normalize --reporter to
            # `tap` so the existing-test gate output is machine-parseable,
            # regardless of which reporter the config declared.
            if re.search(r"--reporter(?:=|\s+)\S+", base_command):
                return re.sub(r"--reporter(?:=|\s+)\S+", "--reporter tap", base_command)
            # No --reporter flag present: insert one before the test path.
            return f"{base_command} --reporter tap"
        return base_command

    def parse(self, stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
        if self.is_mocha:
            return _parse_mocha_tap(stdout, exec_result)
        return _parse_node_tap(stdout, exec_result)

    def lint_command(self, files: list[str]) -> str | None:
        if self.is_mocha:
            # express is plain JavaScript with no tsconfig; skip tsc. The repo
            # has an eslint config, but linting the whole project is out of
            # scope for the gate (which only touches edited files).
            return None
        ts_files = [f for f in (files or []) if f.endswith((".ts", ".tsx"))]
        if not ts_files:
            return None
        # Type-check the whole project (catches type errors introduced in the
        # edited files); tsconfig already scopes the source set.
        return "npx tsc --noEmit -p tsconfig.json"

    def prompt_hints(self) -> dict[str, str]:
        if self.is_mocha:
            example = str(
                self._profile.get(
                    "source_import_example",
                    "const express = require('../lib/express');",
                )
            )
            return {
                "test_runner": "mocha",
                "source_import_example": example,
            }
        example = str(
            self._profile.get(
                "source_import_example",
                'import x from "../middlewares/...";',
            )
        )
        return {
            "test_runner": "node_test",
            "source_import_example": example,
        }

    def count_assertions(self, test_code: str) -> int:
        if not test_code.strip():
            return 0
        patterns = (
            r"\bassert\.\w+\(",
            r"\bassert\(",
            r"\bexpect\(",
            r"\.(?:toBe|toEqual|toThrow|toMatch|toContain)\b",
        )
        return sum(len(re.findall(p, test_code)) for p in patterns)

    def count_test_functions(self, test_code: str) -> int:
        if not test_code.strip():
            return 0
        # Both node:test and mocha use it()/test() as test cases; describe()
        # is a suite, not a test function.
        return len(re.findall(r"\b(?:it|test)\s*\(\s*[\"'`]", test_code))


def _parse_node_tap(stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
    """Parse Node's `node:test` TAP summary (`# tests/pass/fail/...`)."""

    def _grab(label: str) -> int | None:
        # node:test prints a final top-level summary; some setups also print
        # per-subtest summaries. Always take the LAST occurrence — the
        # top-level totals are emitted last.
        matches = re.findall(rf"^#\s*{label}\s+(\d+)\s*$", stdout, re.M)
        return int(matches[-1]) if matches else None

    tests = _grab("tests")
    passed_count = _grab("pass")
    failed_count = _grab("fail")
    cancelled = _grab("cancelled") or 0

    if tests is None and passed_count is None and failed_count is None:
        # No TAP summary at all — runtime/build error before any test ran.
        msg = infer_collection_error(stdout, exec_result.stderr)
        return TestExecutionResult(
            passed=False,
            total=0,
            passed_count=0,
            failed_count=0,
            error_count=1,
            failure_messages=[msg],
            stdout=stdout,
            stderr=exec_result.stderr,
            duration_ms=exec_result.duration_ms,
        )

    passed_count = passed_count or 0
    failed_count = (failed_count or 0) + cancelled
    total = tests if tests is not None else passed_count + failed_count

    failure_messages: list[str] = []
    if failed_count:
        # Surface each `not ok N - <name>` line.
        for m in re.finditer(r"^not ok \d+ - (.+)$", stdout, re.M):
            failure_messages.append(m.group(1).strip())
        if not failure_messages:
            failure_messages.append(f"{failed_count} test(s) failed")

    return TestExecutionResult(
        passed=failed_count == 0 and total > 0,
        total=total,
        passed_count=passed_count,
        failed_count=failed_count,
        error_count=0,
        failure_messages=failure_messages,
        stdout=stdout,
        stderr=exec_result.stderr,
        duration_ms=exec_result.duration_ms,
    )


def _parse_mocha_tap(stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
    """Parse mocha's TAP reporter output (``--reporter tap``).

    mocha TAP emits ``1..N`` then one ``ok N <name>`` / ``not ok N <name>``
    line per test (the separator after the number is a space, optionally a
    ``- ``; a YAML-ish diagnostic block follows under failures). We parse the
    plan line for the total and count ok/not-ok lines for pass/fail.
    """
    plan_match = re.search(r"^1\.\.(\d+)\s*$", stdout, re.M)
    # mocha's TAP writes `ok N description` (no dash); standard TAP writes
    # `ok N - description`. Count by line presence (description optional) so
    # bare `ok N` lines are counted too.
    ok_count = len(re.findall(r"^ok\s+\d+", stdout, re.M))
    not_ok_count = len(re.findall(r"^not ok\s+\d+", stdout, re.M))
    # Extract descriptions for failure messages (separator is ` - ` or space).
    not_ok_names = re.findall(r"^not ok\s+\d+(?:\s*-\s*|\s+)(.+)$", stdout, re.M)

    total = int(plan_match.group(1)) if plan_match else (ok_count + not_ok_count)
    passed_count = ok_count
    failed_count = not_ok_count

    if total == 0 and passed_count == 0 and failed_count == 0:
        # No TAP at all — runtime/build error before any test ran.
        msg = infer_collection_error(stdout, exec_result.stderr)
        return TestExecutionResult(
            passed=False,
            total=0,
            passed_count=0,
            failed_count=0,
            error_count=1,
            failure_messages=[msg],
            stdout=stdout,
            stderr=exec_result.stderr,
            duration_ms=exec_result.duration_ms,
        )

    failure_messages = [name.strip() for name in not_ok_names]
    if failed_count and not failure_messages:
        failure_messages.append(f"{failed_count} test(s) failed")
    if failed_count == 0 and exec_result.return_code not in (0, None):
        # Tests reported green but the process exited non-zero (e.g. a
        # mocha hook threw). Surface the cause.
        failure_messages.append(infer_collection_error(stdout, exec_result.stderr))
        failed_count = max(failed_count, 1)

    return TestExecutionResult(
        passed=failed_count == 0 and total > 0,
        total=total,
        passed_count=passed_count,
        failed_count=failed_count,
        error_count=0,
        failure_messages=failure_messages,
        stdout=stdout,
        stderr=exec_result.stderr,
        duration_ms=exec_result.duration_ms,
    )
