#!/usr/bin/env python3
"""
Module defining a deep neural network performing binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Parameters:
        nx (int): Number of input features.
        layers (list): A list representing the number of nodes
                       in each layer of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Iterate through each layer to initialize weights and biases
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # First hidden layer prev nodes = nx
            # Subsequent layers prev nodes = layers[i-1]
            prev_nodes = nx if i == 0 else layers[i - 1]
            current_nodes = layers[i]

            # He Initialization calculation
            he = np.sqrt(2 / prev_nodes)
            W_key = f'W{i + 1}'
            self.__weights[W_key] = np.random.randn(
                current_nodes, prev_nodes) * he

            # Initialize biases to 0
            b_key = f'b{i + 1}'
            self.__weights[b_key] = np.zeros((current_nodes, 1))

    @property
    def L(self):
        """Retrieves the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Retrieves the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Retrieves the weights dictionary."""
        return self.__weights
