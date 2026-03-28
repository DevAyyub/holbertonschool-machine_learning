#!/usr/bin/env python3
"""Module to calculate a summation without loops."""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from i=1 to n.

    Args:
        n: The stopping condition.

    Returns:
        The integer value of the sum, or None if n is not a valid number.
    """
    if type(n) is not int or n < 1:
        return None

    return (n * (n + 1) * (2 * n + 1)) // 6
