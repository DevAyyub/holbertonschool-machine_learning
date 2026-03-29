#!/usr/bin/env python3
"""Module for matrix multiplication"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication on two 2D matrices
    Args:
        mat1: First 2D matrix (m x n)
        mat2: Second 2D matrix (n x p)
    Returns:
        A new 2D matrix (m x p), or None if incompatible
    """
    if len(mat1[0]) != len(mat2):
        return None

    res_rows = len(mat1)
    res_cols = len(mat2[0])
    inner_dim = len(mat2)

    new_matrix = [[0 for _ in range(res_cols)] for _ in range(res_rows)]

    for i in range(res_rows):
        for j in range(res_cols):
            for k in range(inner_dim):
                new_matrix[i][j] += mat1[i][k] * mat2[k][j]

    return new_matrix
