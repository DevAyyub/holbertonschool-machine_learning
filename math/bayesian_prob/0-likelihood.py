#!/usr/bin/env python3
"""
Module to calculate the likelihood of patient side effects.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given various
    hypothetical probabilities of developing severe side effects.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    def factorial(num):
        f = 1
        for i in range(1, num + 1):
            f *= i
        return f

    n_fact = factorial(n)
    x_fact = factorial(x)
    nx_fact = factorial(n - x)

    combination = n_fact / (x_fact * nx_fact)

    return combination * (P ** x) * ((1 - P) ** (n - x))
