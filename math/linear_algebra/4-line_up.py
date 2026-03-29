#!/usr/bin/env python3
"""Module that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise
    Args:
        arr1: The first array of ints/floats
        arr2: The second array of ints/floats
    Returns:
        A new list with the sums, or None if shapes differ
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
