#!/usr/bin/env python3
"""Batch Normalization Upgraded"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    z = dense(prev)

    mean, variance = tf.nn.moments(z, axes=[0])

    gamma = tf.Variable(tf.ones([n]), name='gamma', trainable=True)
    beta = tf.Variable(tf.zeros([n]), name='beta', trainable=True)

    norm = tf.nn.batch_normalization(
        z,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-7
    )

    return activation(norm)
