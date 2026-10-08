"""Advanced arithmetic operations."""

import math


def power(base: float, exponent: float) -> float:
    """Return base raised to the power of exponent."""
    return base**exponent


def sqrt(x: float) -> float:
    """Return the square root of x. Raises ValueError if x is negative."""
    if x < 0:
        raise ValueError("square root of negative number")
    return math.sqrt(x)


def factorial(n: int) -> int:
    """Return n!. Raises ValueError if n is negative."""
    if n < 0:
        raise ValueError("factorial of negative number")
    return math.factorial(n)


def log(x: float, base: float = math.e) -> float:
    """Return the logarithm of x to the given base. Raises ValueError if x <= 0."""
    if x <= 0:
        raise ValueError("logarithm of non-positive number")
    if base <= 0 or base == 1:
        raise ValueError("invalid logarithm base")
    return math.log(x, base)
