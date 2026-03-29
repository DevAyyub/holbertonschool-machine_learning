#!/usr/bin/env python3
"""
Module containing the Normal class.
"""


class Normal:
    """
    Represents a normal distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            sum_diff_sq = sum([(x - self.mean) ** 2 for x in data])
            variance = sum_diff_sq / len(data)
            self.stddev = float(variance ** 0.5)
