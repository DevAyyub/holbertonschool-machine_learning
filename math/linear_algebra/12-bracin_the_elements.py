#!/usr/bin/env python3
"""Module for element-wise operations using numpy"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication, and division
    Args:
        mat1: The first numpy.ndarray
        mat2: The second numpy.ndarray or a scalar
    Returns:
        A tuple containing (sum, difference, product, quotient)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
