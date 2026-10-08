"""Go language adapter. Per-repo import paths live in ``profiles/<repo>.yaml``."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

from swe_duel.models import ExecutionResult, TestExecutionResult
from swe_duel.sandbox.languages.base import (
    LanguageAdapter,
    infer_collection_error,
    shell_quote,
)
from swe_duel.sandbox.languages.profile_loader import load_repo_profile


class GoAdapter(LanguageAdapter):
    name = "go"

    def __init__(self, repo_name: str | None = None) -> None:
        self.repo_name = (repo_name or "").strip()
        profile = load_repo_profile(self.repo_name)
        self._import_path = str(profile.get("import_path", "") or "")

    def prompt_hints(self) -> dict[str, str]:
        return {"import_path": self._import_path}

    def _go_test_path(self, test_filename: str) -> str:
        """Map a logical label to a Go test file under _swe-duel/ (must end _test.go)."""
        stem = PurePosixPath(test_filename).stem  # drops .py
        stem = re.sub(r"[^0-9A-Za-z_]", "_", stem)
        if not stem.endswith("_test"):
            stem = f"{stem}_test"
        return f"_swe-duel/{stem}.go"

    def prepare_injected(
        self,
        test_filename: str,
        test_code: str,
        target_files: list[str] | None,
    ) -> tuple[dict[str, str], str]:
        path = self._go_test_path(test_filename)
        # Only this one test file lives in _swe-duel/; `go test ./_swe-duel/` compiles
        # and runs the standalone test package it declares.
        command = "go test -buildvcs=false -json ./_swe-duel/"
        return {path: test_code}, command

    def build_existing_command(self, base_command: str, deselects: list[str]) -> str:
        # `go test` has no per-test deselect; ensure -json for reliable counts.
        cmd = base_command
        if "-json" not in cmd.split():
            cmd = cmd.replace("go test", "go test -json", 1)
        return cmd

    def parse(self, stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
        return _parse_go_json(stdout, exec_result)

    def lint_command(self, files: list[str]) -> str | None:
        go_files = [f for f in (files or []) if f.endswith(".go")]
        if not go_files:
            return None
        files_str = " ".join(shell_quote(f) for f in go_files)
        # gofmt must report no diffs; go vet must pass on the affected packages.
        pkgs = sorted({str(PurePosixPath(f).parent) or "." for f in go_files})
        pkg_args = " ".join(f"./{p}/" if p != "." else "." for p in pkgs)
        return f'test -z "$(gofmt -l {files_str})" && go vet -buildvcs=false {pkg_args}'

    def count_assertions(self, test_code: str) -> int:
        if not test_code.strip():
            return 0
        # Go tests assert by failing the test: t.Error/t.Fatal(+f variants),
        # plus testify-style assert./require. calls if present.
        patterns = (
            r"\bt\.Errorf?\b",
            r"\bt\.Fatalf?\b",
            r"\b(?:assert|require)\.\w+\(",
        )
        return sum(len(re.findall(p, test_code)) for p in patterns)

    def count_test_functions(self, test_code: str) -> int:
        if not test_code.strip():
            return 0
        return len(re.findall(r"^\s*func\s+(Test|Example|Benchmark)\w*\s*\(", test_code, re.M))


def _parse_go_json(stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
    """Parse `go test -json` event stream into counts.

    Falls back to the plain `ok`/`FAIL` package-line format when the stream is
    not JSON (e.g. a build failure prints to stderr / plain stdout).
    """
    outcomes: dict[tuple[str, str], str] = {}
    failure_output: dict[tuple[str, str], list[str]] = {}
    saw_json = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        saw_json = True
        action = ev.get("Action")
        test = ev.get("Test")
        pkg = ev.get("Package", "")
        if not test:
            continue
        key = (pkg, test)
        if action in ("pass", "fail", "skip"):
            outcomes[key] = action
        elif action == "output":
            out = ev.get("Output", "")
            if out.strip():
                failure_output.setdefault(key, []).append(out)

    if not saw_json:
        return _parse_go_text(stdout, exec_result)

    passed_count = sum(1 for v in outcomes.values() if v == "pass")
    failed_count = sum(1 for v in outcomes.values() if v == "fail")
    skipped = sum(1 for v in outcomes.values() if v == "skip")
    total = passed_count + failed_count + skipped

    failure_messages: list[str] = []
    for key, outcome in outcomes.items():
        if outcome == "fail":
            pkg, test = key
            tail = "".join(failure_output.get(key, [])[-4:]).strip()
            failure_messages.append(f"{pkg}::{test}: {tail[:200]}" if tail else f"{pkg}::{test}")

    if total == 0:
        # No test functions ran. If the command also failed, surface why.
        if exec_result.return_code not in (0, None):
            failure_messages.append(infer_collection_error(stdout, exec_result.stderr))

    return TestExecutionResult(
        passed=failed_count == 0 and total > 0 and exec_result.return_code in (0, None),
        total=total,
        passed_count=passed_count,
        failed_count=failed_count,
        error_count=0,
        failure_messages=failure_messages,
        stdout=stdout,
        stderr=exec_result.stderr,
        duration_ms=exec_result.duration_ms,
    )


def _parse_go_text(stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
    """Fallback parser for non-JSON `go test` output (build failure etc.)."""
    pkg_ok = len(re.findall(r"^ok\s+\S+", stdout, re.M))
    pkg_fail = len(re.findall(r"^(FAIL|---\s+FAIL)", stdout, re.M))
    failure_messages: list[str] = []
    if pkg_fail or exec_result.return_code not in (0, None):
        failure_messages.append(infer_collection_error(stdout, exec_result.stderr))
    # We cannot count individual tests here; treat any package failure or
    # non-zero exit as failed, a clean run with ≥1 ok package as passed.
    passed = pkg_fail == 0 and pkg_ok > 0 and exec_result.return_code in (0, None)
    total = pkg_ok + pkg_fail
    return TestExecutionResult(
        passed=passed,
        total=total,
        passed_count=pkg_ok,
        failed_count=pkg_fail,
        error_count=0 if passed or total else (1 if exec_result.return_code not in (0, None) else 0),
        failure_messages=failure_messages,
        stdout=stdout,
        stderr=exec_result.stderr,
        duration_ms=exec_result.duration_ms,
    )
