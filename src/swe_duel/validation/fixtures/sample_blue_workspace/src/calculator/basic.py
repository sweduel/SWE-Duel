"""Basic arithmetic operations."""


def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return a minus b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of a and b."""
    return a * b


def divide(a: float, b: float) -> float:
    """Return a divided by b. Raises ZeroDivisionError if b is zero."""
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


def modulo(a: float, b: float) -> float:
    """Return a modulo b. Raises ZeroDivisionError if b is zero."""
    if b == 0:
        raise ZeroDivisionError("modulo by zero")
    return a % b
