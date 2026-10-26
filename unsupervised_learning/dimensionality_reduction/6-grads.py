#!/usr/bin/env python3
"""
Gradients of Y Module
"""
import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y:
    - Y: numpy.ndarray of shape (n, ndim) low dimensional transformation
    - P: numpy.ndarray of shape (n, n) P affinities
    Returns: (dY, Q)
      - dY: numpy.ndarray of shape (n, ndim) gradients of Y
      - Q: numpy.ndarray of shape (n, n) Q affinities of Y
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    # Gradient: sum_j ((P_ij - Q_ij) * (1 + ||y_i - y_j||^2)^(-1) * (y_i - y_j))
    # Note: num_ij = (1 + ||y_i - y_j||^2)^(-1)
    # The requirement explicitly specifies: do not multiply by the scalar 4
    PQ = (P - Q) * num
    dY = np.zeros((n, ndim))

    for i in range(n):
        dY[i] = np.sum((PQ[i, :, None]) * (Y[i] - Y), axis=0)

    return dY, Q
