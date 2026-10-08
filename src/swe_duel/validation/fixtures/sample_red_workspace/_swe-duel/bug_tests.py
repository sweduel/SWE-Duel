"""Bug reproduction tests for the modulo() addition.

These tests FAIL when the bug (truncate-toward-zero) is present and PASS
once modulo() uses Python's % semantics (floor-toward-negative-infinity).
"""

from calculator.basic import modulo


def test_modulo_negative_dividend_matches_python():
    # Python: -5 % 3 == 1. Buggy impl returns -2.
    assert modulo(-5, 3) == 1


def test_modulo_negative_divisor_matches_python():
    # Python: 5 % -3 == -1. Buggy impl returns 2.
    assert modulo(5, -3) == -1
