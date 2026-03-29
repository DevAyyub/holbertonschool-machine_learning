#!/usr/bin/env python3
"""
Module to build a Keras Sequential model
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The keras model
    """
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            # Use input_shape in the first layer since Input class is forbidden
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=reg,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=reg
            ))

        # Add Dropout after every layer except the last one (output layer)
        if i < len(layers) - 1:
            # Keras Dropout takes (1 - keep_prob)
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
