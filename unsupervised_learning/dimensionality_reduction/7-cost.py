#!/usr/bin/env python3
"""
t-SNE Cost Module
"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation:
    - P: numpy.ndarray of shape (n, n) containing P affinities
    - Q: numpy.ndarray of shape (n, n) containing Q affinities
    Returns: C, the cost of the transformation (KL divergence)
    """
    # Clip values to avoid log(0) and division by 0
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    C = np.sum(P * np.log(P / Q))
    return C
