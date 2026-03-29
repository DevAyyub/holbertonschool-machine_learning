#!/usr/bin/env python3
"""Module to add two N-dimensional matrices"""


def add_matrices(mat1, mat2):
    """Adds two matrices of any dimension element-wise
    Args:
        mat1: The first matrix
        mat2: The second matrix
    Returns:
        A new matrix with the sums, or None if shapes differ
    """
    if len(mat1) != len(mat2):
        return None

    if not isinstance(mat1[0], list):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    res = []
    for i in range(len(mat1)):
        if isinstance(mat1[i], list) != isinstance(mat2[i], list):
            return None
        inner = add_matrices(mat1[i], mat2[i])
        if inner is None:
            return None
        res.append(inner)

    return res
