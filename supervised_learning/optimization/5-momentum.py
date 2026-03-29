#!/usr/bin/env python3
"""
Module to update variables using Gradient Descent with Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization
    Args:
        alpha: the learning rate
        beta1: the momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: the previous first moment of var
    Returns:
        The updated variable and the new moment, respectively
    """
    v_new = (beta1 * v) + ((1 - beta1) * grad)
    var_updated = var - (alpha * v_new)
    return var_updated, v_new
