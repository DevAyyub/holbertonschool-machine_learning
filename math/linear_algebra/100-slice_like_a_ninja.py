#!/usr/bin/env python3
"""Module to slice a numpy array along specific axes"""


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes
    Args:
        matrix: The numpy.ndarray to slice
        axes: Dictionary of axis indices and slice tuples
    Returns:
        A new sliced numpy.ndarray
    """
    slc = [slice(*axes.get(i, (None,))) for i in range(matrix.ndim)]
    return matrix[tuple(slc)]
