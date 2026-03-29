#!/usr/bin/env python3
"""
Module containing the Poisson class to represent a Poisson distribution.
"""


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Poisson distribution.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of 'successes'.
        """
        if k < 0:
            return 0

        k = int(k)
        e = 2.7182818285
        lambtha = self.lambtha

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        pmf = ((e ** -lambtha) * (lambtha ** k)) / factorial
        return pmf
