#!/usr/bin/env python3
"""
Module to calculate L2 Regularization Cost using TensorFlow
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization
    Args:
        cost: tensor containing the cost without L2 regularization
        model: Keras model with layers containing L2 regularization
    Returns:
        A tensor containing the total cost for each layer,
        accounting for L2 regularization
    """
    # model.losses contains the regularization losses for each layer
    # We add the scalar cost to the list of losses
    return cost + model.losses
