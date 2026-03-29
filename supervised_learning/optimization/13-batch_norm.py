#!/usr/bin/env python3
"""
Module to perform Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch norm
    Args:
        Z: numpy.ndarray of shape (m, n) to be normalized
        gamma: numpy.ndarray of shape (1, n) containing the scales
        beta: numpy.ndarray of shape (1, n) containing the offsets
        epsilon: small number to avoid division by zero
    Returns:
        The normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_hat = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_hat + beta
