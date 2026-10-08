"""LanguageAdapter ABC and shared helpers used by every adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod

from swe_duel.models import ExecutionResult, TestExecutionResult


class LanguageAdapter(ABC):
    """Strategy object for one source language's test/lint/complexity mechanics."""

    name: str

    # ── injected tests ─────────────────────────────────────

    @abstractmethod
    def prepare_injected(
        self,
        test_filename: str,
        test_code: str,
        target_files: list[str] | None,
    ) -> tuple[dict[str, str], str]:
        """Return (extra_files, command) for running one injected test file.

        `test_filename` is a logical label (e.g. ``test_swe_duel_feature.py``); the
        adapter is free to remap it to a language-appropriate on-disk path.
        """

    def filter_file_overrides(self, file_overrides: dict[str, str]) -> dict[str, str]:
        """Return ``file_overrides`` with build-system files stripped.

        Red agents sometimes edit ``pom.xml``/``build.gradle``/``Makefile``
        to work around perceived environment issues.  The validation gate should
        test source changes, not agent-produced build-tool edits, so adapters
        can drop those files here.  Default: pass through unchanged.
        """
        return dict(file_overrides)

    # ── existing tests ─────────────────────────────────────

    def build_existing_command(self, base_command: str, deselects: list[str]) -> str:
        """Build the existing-test command, applying any deselections.

        Only pytest can express arbitrary nodeid deselection on the command
        line, so the base implementation ignores `deselects`. Python overrides.
        """
        return base_command

    def supports_baseline_deselect(self) -> bool:
        """Whether pre-existing-failure deselection is expressible for this lang."""
        return False

    # ── parsing ────────────────────────────────────────────

    @abstractmethod
    def parse(self, stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
        """Parse a test run's stdout into a TestExecutionResult."""

    # ── lint / type check ──────────────────────────────────

    def lint_command(self, files: list[str]) -> str | None:
        """Return a lint+typecheck command for the given files, or None to skip."""
        return None

    # ── complexity (run against feature_test_code) ─────────

    @abstractmethod
    def count_assertions(self, test_code: str) -> int: ...

    @abstractmethod
    def count_test_functions(self, test_code: str) -> int: ...

    def prompt_hints(self) -> dict[str, str]:
        """Repo-specific values the Red/Blue prompt builder interpolates.

        Default is empty (Python needs none). Language adapters that vary per
        repo (Go import path, Java module/build tool, C header/layout, Node
        runner) override this.
        """
        return {}


def shell_quote(s: str) -> str:
    if not s or any(c in s for c in " \t\n'\""):
        escaped = s.replace("'", "'\\''")
        return f"'{escaped}'"
    return s


def infer_collection_error(stdout: str, stderr: str) -> str:
    """Extract a concise one-liner explaining why 0 tests were collected/run."""
    for stream in (stderr, stdout):
        if not stream:
            continue
        for line in stream.splitlines():
            line = line.strip()
            if not line:
                continue
            if (
                line.startswith("ERROR")
                or line.startswith("ImportError")
                or line.startswith("ModuleNotFoundError")
                or "command not found" in line
                or "no test files" in line.lower()
                or "build failed" in line.lower()
                or "cannot find" in line.lower()
                or "file or directory not found" in line
                or "no tests ran" in line.lower()
                or "no tests to run" in line.lower()
                or "InternalError" in line
                or "panic:" in line
            ):
                return line[:500]
    tail = (stdout or stderr or "").strip().splitlines()[-3:]
    return (" | ".join(tail) or "test runner produced no summary output")[:500]
