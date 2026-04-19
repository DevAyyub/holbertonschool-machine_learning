#!/usr/bin/env python3
"""
DenseNet-121 architecture module for Deep CNNs
"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks.

    Args:
        growth_rate: the growth rate
        compression: the compression factor

    Returns:
        The compiled keras model
    """
    init = K.initializers.HeNormal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Initial Convolution (pre-activation pattern applied to input)
    X = K.layers.BatchNormalization(axis=3)(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=64,
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(X)

    # Dense Block 1 (6 layers) + Transition Layer
    X, nb_filters = dense_block(X, 64, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2 (12 layers) + Transition Layer
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3 (24 layers) + Transition Layer
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4 (16 layers) - No transition layer after the final block
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Classification Layer
    X = K.layers.AveragePooling2D(pool_size=(7, 7))(X)
    X = K.layers.Dense(units=1000,
                       activation='softmax',
                       kernel_initializer=init)(X)

    # Create the model
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
