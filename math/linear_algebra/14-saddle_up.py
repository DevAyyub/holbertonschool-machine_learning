#!/usr/bin/env python3
"""Module for matrix multiplication using numpy"""
import numpy as np


def np_matmul(mat1, mat2):
    """Performs matrix multiplication on two numpy arrays
    Args:
        mat1: The first numpy.ndarray
        mat2: The second numpy.ndarray
    Returns:
        A new numpy.ndarray (the product)
    """
    return np.matmul(mat1, mat2)
