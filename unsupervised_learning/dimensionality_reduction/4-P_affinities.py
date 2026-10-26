#!/usr/bin/env python3
"""
Symmetric P affinities Module
"""
import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a dataset:
    - X: numpy.ndarray of shape (n, d)
    - tol: maximum tolerance allowed for Shannon entropy difference
    - perplexity: perplexity for all Gaussian distributions
    Returns: P, a numpy.ndarray of shape (n, n) of symmetric affinities
    """
    n, _ = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        # Exclude point i from its own distance vector
        Di = np.append(D[i, :i], D[i, i + 1:])
        beta = 1.0
        low = None
        high = None

        Hi, Pi = HP(Di, beta)
        H_diff = Hi - H

        while np.abs(H_diff) > tol:
            if H_diff > 0:
                low = beta
                if high is None:
                    beta = beta * 2.0
                else:
                    beta = (beta + high) / 2.0
            else:
                high = beta
                if low is None:
                    beta = beta / 2.0
                else:
                    beta = (beta + low) / 2.0

            Hi, Pi = HP(Di, beta)
            H_diff = Hi - H

        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    P = (P + P.T) / (2 * n)
    return P
