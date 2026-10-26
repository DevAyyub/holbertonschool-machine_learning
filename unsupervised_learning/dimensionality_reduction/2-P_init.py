#!/usr/bin/env python3
"""
Initialize t-SNE variables Module
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate P affinities in t-SNE:
    - X: numpy.ndarray of shape (n, d)
    - perplexity: perplexity for all Gaussian distributions
    Returns: (D, P, betas, H)
      - D: squared pairwise distance matrix of shape (n, n)
      - P: affinities matrix initialized to 0s of shape (n, n)
      - betas: beta values initialized to 1s of shape (n, 1)
      - H: Shannon entropy for perplexity with base 2
    """
    n, _ = X.shape

    # Squared pairwise distance: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
