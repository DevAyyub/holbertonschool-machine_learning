#!/usr/bin/env python3
"""
Module to calculate L2 Regularization Cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of weights and biases (numpy.ndarrays)
        L: number of layers in the neural network
        m: number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    l2_sum = 0
    for i in range(1, L + 1):
        key = "W{}".format(i)
        l2_sum += np.linalg.norm(weights[key])
    
    # We use np.linalg.norm(weights[key]) which is Frobenius norm by default.
    # Note: The sum of squares is norm squared.
    # However, np.linalg.norm returns the sqrt(sum(squares)).
    # To get sum of squares we can do: np.sum(np.square(weights[key]))
    
    # Redefining sum for clarity using np.sum(np.square())
    l2_cost_total = 0
    for i in range(1, L + 1):
        key = "W{}".format(i)
        l2_cost_total += np.sum(np.square(weights[key]))

    l2_reg = (lambtha / (2 * m)) * l2_cost_total
    return cost + l2_reg
