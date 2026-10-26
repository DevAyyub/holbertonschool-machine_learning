#!/usr/bin/env python3
"""
PCA Module
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset:
    - X: numpy.ndarray of shape (n, d)
    - var: fraction of the variance that the PCA transformation should maintain
    Returns: the weights matrix, W, of shape (d, nd)
    """
    _, s, Vh = np.linalg.svd(X)
    cum_var = np.cumsum(s) / np.sum(s)
    nd = np.argmax(cum_var >= var) + 1
    W = Vh.T[:, :nd]
    return W
