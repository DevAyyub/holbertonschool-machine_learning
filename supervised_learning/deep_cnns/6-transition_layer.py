#!/usr/bin/env python3
"""
Transition Layer module for Deep CNNs
"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    Densely Connected Convolutional Networks.

    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        compression: compression factor for the transition layer

    Returns:
        The output of the transition layer and the number of filters
        within the output, respectively
    """
    init = K.initializers.HeNormal(seed=0)

    # Calculate compressed number of filters (DenseNet-C)
    nb_filters = int(nb_filters * compression)

    # Pre-activation: Batch Normalization -> ReLU
    Y = K.layers.BatchNormalization(axis=3)(X)
    Y = K.layers.Activation('relu')(Y)

    # 1x1 Convolution for compression
    Y = K.layers.Conv2D(filters=nb_filters,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=init)(Y)

    # 2x2 Average Pooling for spatial reduction
    Y = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  padding='same')(Y)

    return Y, nb_filters
