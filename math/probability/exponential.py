#!/usr/bin/env python3
"""
Module containing the Exponential class.
"""


class Exponential:
    """
    Represents an exponential distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential distribution.
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

            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period.
        """
        if x < 0:
            return 0

        e = 2.7182818285
        lambtha = self.lambtha

        return lambtha * (e ** (-lambtha * x))
