"""Tests for advanced arithmetic operations."""

import math

import pytest
from calculator.advanced import power, sqrt, factorial, log


class TestPower:
    def test_integer_exponent(self):
        assert power(2, 3) == 8

    def test_zero_exponent(self):
        assert power(5, 0) == 1

    def test_negative_exponent(self):
        assert power(2, -1) == 0.5


class TestSqrt:
    def test_perfect_square(self):
        assert sqrt(9) == 3.0

    def test_non_perfect(self):
        assert sqrt(2) == pytest.approx(1.4142135623730951)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="negative"):
            sqrt(-1)


class TestFactorial:
    def test_zero(self):
        assert factorial(0) == 1

    def test_positive(self):
        assert factorial(5) == 120

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="negative"):
            factorial(-1)


class TestLog:
    def test_natural(self):
        assert log(math.e) == pytest.approx(1.0)

    def test_base_10(self):
        assert log(100, 10) == pytest.approx(2.0)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="non-positive"):
            log(0)
