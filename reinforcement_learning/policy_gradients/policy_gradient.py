#!/usr/bin/env python3
"""
Module that implements Policy Gradient functions for reinforcement learning.
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix using softmax.

    Args:
        matrix (np.ndarray): The state matrix of shape (1, n) or (m, n).
        weight (np.ndarray): The weight matrix of shape (n, a).

    Returns:
        np.ndarray: The softmax policy probability distribution.
    """
    exp = np.exp(matrix @ weight)
    return exp / np.sum(exp)
