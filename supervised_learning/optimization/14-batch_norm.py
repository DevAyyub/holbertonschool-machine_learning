#!/usr/bin/env python3
"""
Module to create a batch normalization layer in TensorFlow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function to be used on the output of the layer
    Returns:
        A tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n, 
        kernel_initializer=init, 
        use_bias=False
    )
    z = layer(prev)

    batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-7)
    z_bn = batch_norm(z)

    if activation is None:
        return z_bn
    return activation(z_bn)
