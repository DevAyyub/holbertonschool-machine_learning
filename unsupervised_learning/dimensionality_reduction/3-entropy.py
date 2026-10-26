#!/usr/bin/env python3
"""
Shannon entropy and P affinities Module
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a point:
    - Di: numpy.ndarray of shape (n - 1,) pairwise distances
    - beta: float or numpy.ndarray containing the beta value for Gaussian
    Returns: (Hi, Pi)
      - Hi: Shannon entropy of the points
      - Pi: numpy.ndarray of shape (n - 1,) P affinities
    """
    P = np.exp(-Di * beta)
    sum_P = np.sum(P)

    Pi = P / sum_P

    Hi = (beta * np.sum(Di * Pi) + np.log(sum_P)) / np.log(2)

    return Hi, Pi
