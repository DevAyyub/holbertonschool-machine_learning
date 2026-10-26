#!/usr/bin/env python3
"""
Shannon entropy and P affinities Module
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a point:
    - Di: numpy.ndarray of shape (n - 1,) pairwise distances
    - beta: numpy.ndarray of shape (1,) beta value for Gaussian
    Returns: (Hi, Pi)
      - Hi: Shannon entropy of the points
      - Pi: numpy.ndarray of shape (n - 1,) P affinities
    """
    # Numerator of P: exp(-beta * Di)
    P = np.exp(-Di * beta[0])
    sum_P = np.sum(P)

    # Normalized affinities Pi
    Pi = P / sum_P

    # Shannon entropy with base 2:
    # H = -sum(Pi * log2(Pi)) = sum(beta * Di * Pi) + log2(sum_P) (in nats/bits)
    # Using H = sum(beta * Di * Pi) + log(sum_P) converted to base 2
    # Or standard definition using natural log converted to base 2:
    Hi = (beta[0] * np.sum(Di * Pi) + np.log(sum_P)) / np.log(2)

    return Hi, Pi
