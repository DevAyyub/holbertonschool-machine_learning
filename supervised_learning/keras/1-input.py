#!/usr/bin/env python3
"""
Module to build a Keras model using the Functional API
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library (Functional API)

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The keras model
    """
    # Define the input layer
    inputs = K.Input(shape=(nx,))

    # Initialize the L2 regularizer
    reg = K.regularizers.l2(lambtha)

    # Set the first layer to take the inputs
    x = inputs

    for i in range(len(layers)):
        # Create each Dense layer and connect it to the previous output 'x'
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=reg
        )(x)

        # Add Dropout after every layer except the last one
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    # Create the model by specifying inputs and the final output 'x'
    model = K.Model(inputs=inputs, outputs=x)

    return model
