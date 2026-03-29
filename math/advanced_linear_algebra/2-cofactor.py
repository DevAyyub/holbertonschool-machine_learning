#!/usr/bin/env python3
"""Module to calculate the cofactor matrix of a matrix"""


def determinant(matrix):
    """Private helper to calculate determinant for cofactors"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    det = 0
    for j in range(n):
        submatrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(submatrix)
    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix
    Args:
        matrix: list of lists whose cofactor matrix should be calculated
    Returns: the cofactor matrix of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            submatrix = [row[:j] + row[j+1:] for row in
                         (matrix[:i] + matrix[i+1:])]
            minor_val = determinant(submatrix)
            row_cofactors.append(minor_val * ((-1) ** (i + j)))
        cofactor_matrix.append(row_cofactors)

    return cofactor_matrix
