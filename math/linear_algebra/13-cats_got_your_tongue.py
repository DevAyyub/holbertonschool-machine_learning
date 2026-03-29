#!/usr/bin/env python3
"""Module to concatenate two numpy arrays"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis
    Args:
        mat1: The first numpy.ndarray
        mat2: The second numpy.ndarray
        axis: The axis along which to concatenate
    Returns:
        A new numpy.ndarray (concatenated)
    """
    return np.concatenate((mat1, mat2), axis=axis)
