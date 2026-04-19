#!/usr/bin/env python3
"""
Projection Block module for Deep CNNs
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015).
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # Main Path
    X = K.layers.Conv2D(filters=F11,
                        kernel_size=(1, 1),
                        strides=(s, s),
                        padding='same',
                        kernel_initializer=init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F3,
                        kernel_size=(3, 3),
                        padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F12,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut Path
    X_shortcut = K.layers.Conv2D(filters=F12,
                                 kernel_size=(1, 1),
                                 strides=(s, s),
                                 padding='same',
                                 kernel_initializer=init)(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Add shortcut value to main path
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
