#!/usr/bin/env python3
"""
Module to perform one-hot decoding.
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Parameters:
    one_hot (numpy.ndarray): One-hot encoded matrix with shape (classes, m).

    Returns:
    A numpy.ndarray with shape (m,) containing the labels, or None on failure.
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    # axis=0 looks at the rows within each column to find the index of the 1
    return np.argmax(one_hot, axis=0)
