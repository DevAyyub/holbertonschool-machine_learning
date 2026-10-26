#!/usr/bin/env python3
"""
Q affinities Module
"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities:
    - Y: numpy.ndarray of shape (n, ndim) low dimensional representation
    Returns: (Q, num)
      - Q: numpy.ndarray of shape (n, n) containing the Q affinities
      - num: numpy.ndarray of shape (n, n) containing the numerator
    """
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)

    num = 1.0 / (1.0 + D)
    np.fill_diagonal(num, 0.0)

    Q = num / np.sum(num)

    return Q, num
