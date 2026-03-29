#!/usr/bin/env python3
"""
Module to create a TensorFlow layer with L2 Regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow with L2 regularization
    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function used on the layer
        lambtha: L2 regularization parameter
    Returns: the output of the new layer
    """
    # Define the L2 regularizer
    regularizer = tf.keras.regularizers.L2(lambtha)

    # Define the specific initializer required
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    # Create the Dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
