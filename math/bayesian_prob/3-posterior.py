#!/usr/bin/env python3
"""
Module to calculate the posterior probability of patient side effects.
"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for various hypothetical 
    probabilities of developing severe side effects.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    import math
    # Intersection = Likelihood * Prior
    fact_n = math.factorial(n)
    fact_x = math.factorial(x)
    fact_nx = math.factorial(n - x)
    comb = fact_n / (fact_x * fact_nx)
    likelihood = comb * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihood * Pr

    # Marginal = sum of intersections
    marginal = np.sum(intersection)

    # Posterior = Intersection / Marginal
    return intersection / marginal
