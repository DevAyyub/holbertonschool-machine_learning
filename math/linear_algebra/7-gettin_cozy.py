#!/usr/bin/env python3
"""Module that concatenates two 2D matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along a specific axis
    Args:
        mat1: The first 2D matrix
        mat2: The second 2D matrix
        axis: The axis along which to concatenate (0 for rows, 1 for cols)
    Returns:
        A new 2D matrix, or None if the shapes are incompatible
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
