#!/usr/bin/env python3
"""
Module to perform one-hot encoding.
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Parameters:
    Y (numpy.ndarray): Numeric class labels with shape (m,).
    classes (int): Maximum number of classes found in Y.

    Returns:
    A one-hot encoding of Y with shape (classes, m), or None on failure.
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= np.amax(Y):
        return None

    m = Y.shape[0]
    # Initialize a matrix of zeros
    one_hot = np.zeros((classes, m))

    # Use numpy indexing to set the correct row for each column to 1.0
    # np.arange(m) selects the column, Y selects the row
    one_hot[Y, np.arange(m)] = 1.0

    return one_hot
