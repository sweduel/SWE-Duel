"""Complexity and non-triviality checks for Red challenges."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from swe_duel.config import ArenaConfig
from swe_duel.sandbox.diff_utils import compute_diff_stats

if TYPE_CHECKING:
    from swe_duel.sandbox.languages import LanguageAdapter


def count_pytest_assertions(test_code: str) -> int:
    """Count `assert` statements plus `pytest.raises` context managers."""
    if not test_code.strip():
        return 0
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        # Fallback: simple text scan
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
        elif isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and _is_pytest_raises(call.func):
                    count += 1
        elif isinstance(node, ast.AsyncWith):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and _is_pytest_raises(call.func):
                    count += 1
    return count


def _is_pytest_raises(func: ast.expr) -> bool:
    if isinstance(func, ast.Attribute):
        return (
            func.attr == "raises"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
        )
    return False


def count_test_functions(test_code: str) -> int:
    """Count top-level and nested `def test_*` definitions."""
    if not test_code.strip():
        return 0
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return sum(1 for line in test_code.splitlines() if line.lstrip().startswith("def test_"))

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                count += 1
    return count


def check_thresholds(
    pr_diff: str,
    feature_test_code: str,
    config: ArenaConfig,
    adapter: "LanguageAdapter | None" = None,
) -> tuple[bool, str]:
    """Check diff size, assertion count, and test-function count thresholds.

    Assertion / test-function counting is language-specific: when an `adapter`
    is supplied it owns the counting (Go ``t.Error``/``t.Fatal``, Node
    ``assert``/``expect``/``it``/``test``); otherwise the Python/pytest counters
    are used for backwards compatibility.
    """
    stats = compute_diff_stats(pr_diff)
    if adapter is not None:
        assertions = adapter.count_assertions(feature_test_code)
        functions = adapter.count_test_functions(feature_test_code)
    else:
        assertions = count_pytest_assertions(feature_test_code)
        functions = count_test_functions(feature_test_code)

    gates = config.red_gates
    problems: list[str] = []

    if stats.lines_added < gates.min_diff_lines:
        problems.append(
            f"lines_added={stats.lines_added} < min_diff_lines={gates.min_diff_lines}"
        )
    if assertions < gates.min_test_assertions:
        problems.append(
            f"assertions={assertions} < min_test_assertions={gates.min_test_assertions}"
        )
    if functions < gates.min_test_functions:
        problems.append(
            f"test_functions={functions} < min_test_functions={gates.min_test_functions}"
        )

    if problems:
        return False, "; ".join(problems)

    return True, (
        f"lines_added={stats.lines_added}, assertions={assertions}, "
        f"test_functions={functions}"
    )
