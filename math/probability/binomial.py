#!/usr/bin/env python3
"""
Module containing the Binomial class.
"""


class Binomial:
    """
    Represents a binomial distribution.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial distribution.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            sum_diff_sq = sum([(x - mean) ** 2 for x in data])
            variance = sum_diff_sq / len(data)

            p_estimated = 1 - (variance / mean)
            self.n = int(round(mean / p_estimated))
            self.p = float(mean / self.n)
