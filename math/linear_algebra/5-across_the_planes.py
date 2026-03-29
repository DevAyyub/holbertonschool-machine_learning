#!/usr/bin/env python3
"""Module that adds two 2D matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise
    Args:
        mat1: The first 2D matrix
        mat2: The second 2D matrix
    Returns:
        A new 2D matrix with the sums, or None if shapes differ
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
