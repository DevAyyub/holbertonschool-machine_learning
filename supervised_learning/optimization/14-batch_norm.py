#!/usr/bin/env python3
"""Batch Normalization Upgraded"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    """
    # Base Dense layer with VarianceScaling initializer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    
    # Get the linear output of the layer
    z = dense(prev)

    # Calculate mean and variance along the batch axis (0)
    mean, variance = tf.nn.moments(z, axes=[0])

    # Initialize trainable parameters gamma (1s) and beta (0s)
    gamma = tf.Variable(tf.ones([n]), name='gamma', trainable=True)
    beta = tf.Variable(tf.zeros([n]), name='beta', trainable=True)

    # Apply normalization and scale/shift (y = gamma * norm + beta)
    # Using epsilon = 1e-7 as per requirements
    norm = tf.nn.batch_normalization(
        z,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-7
    )

    # Return the activated output
    return activation(norm)
