#!/usr/bin/env python3
"""Module that transposes a 2D matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix
    Args:
        matrix: The 2D matrix to transpose
    Returns:
        A new matrix (the transpose)
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
