#!/usr/bin/env python3
"""
Dense Block module for Deep CNNs
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in
    Densely Connected Convolutional Networks.

    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        The concatenated output of each layer within the Dense Block and
        the number of filters within the concatenated outputs, respectively
    """
    init = K.initializers.HeNormal(seed=0)

    for i in range(layers):
        # Bottleneck Layer (1x1 Convolution)
        # Pre-activation: BN -> ReLU -> Conv
        Y = K.layers.BatchNormalization(axis=3)(X)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(filters=4 * growth_rate,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=init)(Y)

        # Standard Layer (3x3 Convolution)
        # Pre-activation: BN -> ReLU -> Conv
        Y = K.layers.BatchNormalization(axis=3)(Y)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer=init)(Y)

        # Concatenate the original input with the new feature maps
        X = K.layers.Concatenate(axis=3)([X, Y])

        # Update the total number of filters
        nb_filters += growth_rate

    return X, nb_filters
