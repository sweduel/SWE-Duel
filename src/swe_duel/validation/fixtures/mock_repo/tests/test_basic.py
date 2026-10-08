"""Tests for basic arithmetic operations."""

import pytest
from calculator.basic import add, subtract, multiply, divide


class TestAdd:
    def test_positive(self):
        assert add(2, 3) == 5

    def test_negative(self):
        assert add(-1, -2) == -3


class TestSubtract:
    def test_positive(self):
        assert subtract(5, 3) == 2

    def test_negative_result(self):
        assert subtract(3, 5) == -2


class TestMultiply:
    def test_positive(self):
        assert multiply(3, 4) == 12

    def test_by_zero(self):
        assert multiply(5, 0) == 0


class TestDivide:
    def test_exact(self):
        assert divide(10, 2) == 5.0

    def test_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            divide(1, 0)
