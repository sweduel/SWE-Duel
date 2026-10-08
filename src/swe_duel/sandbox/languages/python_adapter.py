"""Python / pytest language adapter."""

from __future__ import annotations

import json
import re

from swe_duel.models import ExecutionResult, TestExecutionResult
from swe_duel.sandbox.languages.base import (
    LanguageAdapter,
    infer_collection_error,
    shell_quote,
)


class PythonAdapter(LanguageAdapter):
    name = "python"

    def prepare_injected(
        self,
        test_filename: str,
        test_code: str,
        target_files: list[str] | None,
    ) -> tuple[dict[str, str], str]:
        command = f"python -m pytest {test_filename} -x -q --tb=short"
        return {test_filename: test_code}, command

    def build_existing_command(self, base_command: str, deselects: list[str]) -> str:
        if not deselects:
            return base_command
        flags = " ".join(f"--deselect {shell_quote(n)}" for n in deselects)
        return f"{base_command} {flags}"

    def supports_baseline_deselect(self) -> bool:
        return True

    def parse(self, stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
        try:
            report = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return self._parse_text(stdout, exec_result)

        summary = report.get("summary", {})
        total = summary.get("total", 0)
        passed_count = summary.get("passed", 0)
        failed_count = summary.get("failed", 0)
        error_count = summary.get("error", 0)

        failure_messages: list[str] = []
        for test in report.get("tests", []):
            if test.get("outcome") in ("failed", "error"):
                call_info = test.get("call", {})
                crash = call_info.get("crash", {})
                msg = crash.get("message", test.get("outcome", "unknown"))
                failure_messages.append(f"{test['nodeid']}: {msg}")

        if total == 0 and exec_result.return_code not in (0, None):
            error_count = max(error_count, 1)
            failure_messages.append(infer_collection_error(stdout, exec_result.stderr))

        return TestExecutionResult(
            passed=failed_count == 0 and error_count == 0 and total > 0,
            total=total,
            passed_count=passed_count,
            failed_count=failed_count,
            error_count=error_count,
            failure_messages=failure_messages,
            stdout=stdout,
            stderr=exec_result.stderr,
            duration_ms=exec_result.duration_ms,
        )

    def _parse_text(self, stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
        passed_count = failed_count = error_count = 0
        for m in re.finditer(r"(\d+) (passed|failed|errors?)", stdout):
            count, label = int(m.group(1)), m.group(2)
            if label == "passed":
                passed_count = count
            elif label == "failed":
                failed_count = count
            elif label.startswith("error"):
                error_count = count
        total = passed_count + failed_count + error_count

        failure_messages = [m.group(1).strip() for m in re.finditer(r"FAILED (.+)", stdout)]

        if total == 0 and exec_result.return_code not in (0, None):
            error_count = max(error_count, 1)
            failure_messages.append(infer_collection_error(stdout, exec_result.stderr))

        return TestExecutionResult(
            passed=failed_count == 0 and error_count == 0 and total > 0,
            total=total,
            passed_count=passed_count,
            failed_count=failed_count,
            error_count=error_count,
            failure_messages=failure_messages,
            stdout=stdout,
            stderr=exec_result.stderr,
            duration_ms=exec_result.duration_ms,
        )

    def lint_command(self, files: list[str]) -> str | None:
        if not files:
            return None
        files_str = " ".join(shell_quote(f) for f in files)
        return f"ruff check {files_str} && mypy {files_str} --ignore-missing-imports"

    def count_assertions(self, test_code: str) -> int:
        import ast

        if not test_code.strip():
            return 0
        try:
            tree = ast.parse(test_code)
        except SyntaxError:
            count = 0
            for line in test_code.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("assert ") or stripped.startswith("assert("):
                    count += 1
                if "pytest.raises" in stripped:
                    count += 1
            return count

        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                count += 1
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    call = item.context_expr
                    if isinstance(call, ast.Call) and _is_pytest_raises(call.func):
                        count += 1
        return count

    def count_test_functions(self, test_code: str) -> int:
        import ast

        if not test_code.strip():
            return 0
        try:
            tree = ast.parse(test_code)
        except SyntaxError:
            return sum(
                1 for line in test_code.splitlines() if line.lstrip().startswith("def test_")
            )
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    count += 1
        return count


def _is_pytest_raises(func: object) -> bool:
    import ast

    if isinstance(func, ast.Attribute):
        return (
            func.attr == "raises"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
        )
    return False
