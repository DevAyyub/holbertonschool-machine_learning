#!/usr/bin/env python3
"""Module to calculate the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a numpy.ndarray
    Args:
        matrix: numpy.ndarray of shape (n, n)
    Returns:
        String representing the definiteness, or None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    try:
        # Check if 2D, square, and symmetric
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or \
           not np.all(matrix == matrix.T):
            return None

        evals = np.linalg.eigvals(matrix)

        pos = np.any(evals > 1e-10)
        neg = np.any(evals < -1e-10)
        # Using a small epsilon for 0 to handle floating point noise
        zero = np.any(np.isclose(evals, 0, atol=1e-10))

        if pos and not neg and not zero:
            return "Positive definite"
        if pos and not neg and zero:
            return "Positive semi-definite"
        if neg and not pos and not zero:
            return "Negative definite"
        if neg and not pos and zero:
            return "Negative semi-definite"
        if pos and neg:
            return "Indefinite"

        return None
    except Exception:
        return None
