#!/usr/bin/env python3
"""
Module to shuffle data points in two matrices
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle
        Y: numpy.ndarray of shape (m, ny) to shuffle
    Returns:
        The shuffled X and Y matrices
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
