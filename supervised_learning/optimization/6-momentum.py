#!/usr/bin/env python3
"""
Module to create a momentum optimizer in TensorFlow
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization in TensorFlow
    Args:
        alpha: the learning rate
        beta1: the momentum weight
    Returns:
        The optimizer (tf.keras.optimizers.SGD)
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
