"""Feature tests for the modulo() addition to calculator.basic."""

import pytest

from calculator.basic import modulo


def test_modulo_positive():
    assert modulo(10, 3) == 1


def test_modulo_exact_division():
    assert modulo(9, 3) == 0


def test_modulo_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        modulo(5, 0)
