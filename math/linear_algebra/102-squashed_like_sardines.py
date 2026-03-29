#!/usr/bin/env python3
"""Module to concatenate two N-dimensional matrices"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis
    Args:
        mat1: The first matrix
        mat2: The second matrix
        axis: The axis along which to concatenate
    Returns:
        A new matrix, or None if shapes are incompatible
    """
    def get_shape(matrix):
        """Helper to get the shape of a nested list"""
        shape = []
        while isinstance(matrix, list):
            shape.append(len(matrix))
            matrix = matrix[0] if len(matrix) > 0 else None
        return shape

    def deep_copy(matrix):
        """Helper to create a deep copy of a nested list"""
        if not isinstance(matrix, list):
            return matrix
        return [deep_copy(item) for item in matrix]

    if axis == 0:
        if get_shape(mat1)[1:] != get_shape(mat2)[1:]:
            return None
        return deep_copy(mat1) + deep_copy(mat2)

    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if len(mat1) != len(mat2):
        return None

    res = []
    for i in range(len(mat1)):
        merged = cat_matrices(mat1[i], mat2[i], axis - 1)
        if merged is None:
            return None
        res.append(merged)

    return res
