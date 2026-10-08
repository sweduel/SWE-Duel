"""Phase 7: complexity helper tests."""

from __future__ import annotations

from swe_duel.config import ArenaConfig, RedGatesConfig
from swe_duel.validation.complexity import (
    check_thresholds,
    count_pytest_assertions,
    count_test_functions,
)


_FEATURE_TEST_CODE = """\
import pytest
from calculator.basic import modulo


def test_modulo_basic():
    assert modulo(10, 3) == 1
    assert modulo(9, 3) == 0
    assert modulo(-5, 3) == 1


def test_modulo_raises_on_zero():
    with pytest.raises(ZeroDivisionError):
        modulo(1, 0)


def test_modulo_negative_divisor():
    assert modulo(7, -3) == -2
"""


_TWENTY_LINE_DIFF = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,1 +1,21 @@
-x = 1
+x = 1
+def added_0():
+    return 0
+def added_1():
+    return 1
+def added_2():
+    return 2
+def added_3():
+    return 3
+def added_4():
+    return 4
+def added_5():
+    return 5
+def added_6():
+    return 6
+def added_7():
+    return 7
+def added_8():
+    return 8
+def added_9():
+    return 9
"""


_SMALL_DIFF = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,1 +1,2 @@
 x = 1
+y = 2
"""


def _config(
    min_diff_lines: int = 10,
    min_test_assertions: int = 2,
    min_test_functions: int = 1,
) -> ArenaConfig:
    return ArenaConfig(
        red_gates=RedGatesConfig(
            min_diff_lines=min_diff_lines,
            min_test_assertions=min_test_assertions,
            min_test_functions=min_test_functions,
        )
    )


def test_count_pytest_assertions(record):
    code = """\
import pytest

def test_a():
    assert 1 == 1
    assert 2 == 2
    assert 3 == 3
    assert 4 == 4
    assert 5 == 5
    with pytest.raises(ValueError):
        raise ValueError()
"""
    count = count_pytest_assertions(code)
    record("count", count)
    assert count == 6


def test_count_test_functions(record):
    code = """\
def test_one():
    pass

def test_two():
    pass

def helper():
    pass

def test_three():
    pass

def test_four():
    pass
"""
    count = count_test_functions(code)
    record("count", count)
    assert count == 4


def test_check_thresholds_pass(record):
    config = _config(min_diff_lines=10, min_test_assertions=2, min_test_functions=1)
    passed, message = check_thresholds(_TWENTY_LINE_DIFF, _FEATURE_TEST_CODE, config)
    record("passed", passed)
    record("message", message)
    assert passed


def test_check_thresholds_fail_lines(record):
    config = _config(min_diff_lines=10)
    passed, message = check_thresholds(_SMALL_DIFF, _FEATURE_TEST_CODE, config)
    record("passed", passed)
    record("message", message)
    assert not passed
    assert "min_diff_lines" in message


def test_check_thresholds_fail_assertions(record):
    code = """\
def test_one():
    assert 1 == 1
"""
    config = _config(min_diff_lines=10, min_test_assertions=3)
    passed, message = check_thresholds(_TWENTY_LINE_DIFF, code, config)
    record("passed", passed)
    record("message", message)
    assert not passed
    assert "min_test_assertions" in message
