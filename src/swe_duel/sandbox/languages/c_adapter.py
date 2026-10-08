"""ANSI-C (and C++-via-profile) language adapter.

Per-repo build profiles (glob vs cmake, includes, link flags, headers,
optional ``compiler``/``std``/``test_ext`` for C++ libraries like simdjson)
live in ``profiles/<repo>.yaml``. Optional ``prompt_hints`` keys in the
profile are merged into ``prompt_hints()``.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from swe_duel.models import ExecutionResult, TestExecutionResult
from swe_duel.sandbox.languages.base import (
    LanguageAdapter,
    infer_collection_error,
    shell_quote,
)
from swe_duel.sandbox.languages.profile_loader import load_repo_profile


class CAdapter(LanguageAdapter):
    """Adapter for ANSI-C projects.

    A harness-injected C test is a standalone program with its own `main`.
    The adapter compiles it together with the project's non-test `.c` source
    files, then executes the resulting binary.

    Per-repo source layout is captured in each repo's YAML profile: cJSON keeps
    its ``*.c`` at the workspace root, while libexpat ships sources under
    ``expat/lib/`` and rebuilds via cmake. Missing profile preserves a simple
    root-glob default.
    """

    name = "c"

    def __init__(self, repo_name: str | None = None) -> None:
        self.repo_name = (repo_name or "").strip()
        self._profile = load_repo_profile(self.repo_name)

    def _c_test_path(self, test_filename: str) -> str:
        stem = PurePosixPath(test_filename).stem
        stem = re.sub(r"[^0-9A-Za-z_]", "_", stem)
        # Default `.c`; C++ repos (e.g. simdjson) set `test_ext: cpp` so the
        # injected file matches the `g++` compile path.
        ext = str(self._profile.get("test_ext", "c")).lstrip(".")
        return f"_swe-duel/{stem}.{ext}"

    def _compile_prefix(self) -> str:
        """Return ``{compiler} -std={std}`` honoring per-repo profile knobs.

        Pure-C repos keep the historical ``gcc -std=c99`` default; C++-adjacent
        libraries (simdjson) set ``compiler: g++`` / ``std: c++17``.
        """
        compiler = str(self._profile.get("compiler", "gcc"))
        std = str(self._profile.get("std", "c99"))
        return f"{compiler} -std={std}"

    def prepare_injected(
        self,
        test_filename: str,
        test_code: str,
        target_files: list[str] | None,
    ) -> tuple[dict[str, str], str]:
        path = self._c_test_path(test_filename)
        stem = PurePosixPath(path).stem
        binary = f"_swe-duel/{stem}"
        include_flags = str(self._profile.get("include_flags", "-I."))
        compiler = self._compile_prefix()
        method = str(self._profile.get("build_method", "glob"))
        if method == "cmake":
            cmake_src = str(self._profile.get("cmake_source_root", "."))
            cmake_target = str(self._profile.get("cmake_target", ""))
            cmake_args = str(self._profile.get("cmake_extra_args", ""))
            build_dir = str(self._profile.get("cmake_build_dir", "/tmp/cbuild"))
            link_flags = str(self._profile.get("link_flags", "-lm"))
            target_clause = f"--target {cmake_target} " if cmake_target else ""
            # Rebuild the project's static library from (possibly agent-
            # modified) source, then compile + link the injected test against
            # it. Redirect cmake's own chatter so the adapter sees the test
            # binary's output.
            command = (
                f"cmake -S {cmake_src} -B {build_dir} {cmake_args} >/dev/null 2>&1 "
                f"&& cmake --build {build_dir} {target_clause}>/dev/null 2>&1 "
                f"&& {compiler} {include_flags} -o {binary} {path} {link_flags} "
                f"&& {binary}"
            )
            return {path: test_code}, command

        # glob path (cjson-style)
        explicit = self._profile.get("source_files")
        test_basename = PurePosixPath(path).name
        if explicit:
            sources = str(explicit)
        else:
            source_dir = str(self._profile.get("source_dir", "."))
            exclude_glob = str(self._profile.get("exclude_glob", "test*.c"))
            maxdepth = self._profile.get("maxdepth", 1)
            exclude_clause = f"! -name {exclude_glob} " if exclude_glob else ""
            depth_clause = f"-maxdepth {maxdepth} " if maxdepth else ""
            sources = (
                f"$(find {source_dir} {depth_clause}-type f -name '*.c' "
                f"{exclude_clause}! -name {shell_quote(test_basename)} "
                f"| tr '\\n' ' ')"
            )
        command = f"{compiler} {include_flags} -o {binary} {path} {sources} -lm && {binary}"
        return {path: test_code}, command

    def prompt_hints(self) -> dict[str, str]:
        header = str(self._profile.get("header", ""))
        hints: dict[str, str] = {"header": header}
        # Optional free-form extras declared in the YAML profile.
        extra = self._profile.get("prompt_hints")
        if isinstance(extra, dict):
            for key, value in extra.items():
                hints[str(key)] = str(value)
        elif header:
            hints["include_directive"] = f'#include "{header}"'
        return hints

    def parse(self, stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
        return _parse_c_run(stdout, exec_result)

    def count_assertions(self, test_code: str) -> int:
        if not test_code.strip():
            return 0
        # Standard C assert(...).
        return len(re.findall(r"\bassert\s*\(", test_code))

    def count_test_functions(self, test_code: str) -> int:
        if not test_code.strip():
            return 0
        return len(
            re.findall(
                r"^\s*(?:void|int|static\s+(?:void|int))\s+test_\w+\s*\(",
                test_code,
                re.M,
            )
        )


def _parse_c_run(stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
    """Treat a compiled C test program as a single testable unit.

    assert() failures abort the binary, yielding a non-zero exit code.
    """
    passed = exec_result.return_code in (0, None)
    failure_messages: list[str] = []
    if not passed:
        failure_messages.append(infer_collection_error(stdout, exec_result.stderr))

    return TestExecutionResult(
        passed=passed,
        total=1,
        passed_count=1 if passed else 0,
        failed_count=0 if passed else 1,
        error_count=0,
        failure_messages=failure_messages,
        stdout=stdout,
        stderr=exec_result.stderr,
        duration_ms=exec_result.duration_ms,
    )
