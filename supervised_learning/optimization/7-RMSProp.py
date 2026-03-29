#!/usr/bin/env python3
"""
Module to update variables using RMSProp optimization
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm
    Args:
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        s: the previous second moment of var
    Returns:
        The updated variable and the new moment, respectively
    """
    s_new = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var_updated = var - alpha * (grad / (np.sqrt(s_new) + epsilon))
    return var_updated, s_new
