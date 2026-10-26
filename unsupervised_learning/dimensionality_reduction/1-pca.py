#!/usr/bin/env python3
"""
PCA v2 Module
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset:
    - X: numpy.ndarray of shape (n, d)
    - ndim: new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim)
    """
    X_m = X - np.mean(X, axis=0)
    _, _, Vh = np.linalg.svd(X_m)
    W = Vh.T[:, :ndim]
    T = np.matmul(X_m, W)
    return T
