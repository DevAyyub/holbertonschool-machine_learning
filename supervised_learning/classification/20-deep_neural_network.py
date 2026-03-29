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
        """Initializes the deep neural network."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Iterate through every layer to initialize weights and biases
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_nodes = nx if i == 0 else layers[i - 1]
            current_nodes = layers[i]

            he = np.sqrt(2 / prev_nodes)
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)

            self.__weights[W_key] = np.random.randn(
                current_nodes, prev_nodes) * he
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

    def forward_prop(self, X):
        """Calculates the forward propagation."""
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A{}".format(i)] = A

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        m = Y.shape[1]
        loss_1 = Y * np.log(A)
        loss_2 = (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(loss_1 + loss_2)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network predictions.

        Parameters:
        X (numpy.ndarray): Array with shape (nx, m) input data.
        Y (numpy.ndarray): Array with shape (1, m) correct labels.

        Returns:
        The prediction and the cost of the network, respectively.
        """
        # Run forward propagation; only the final output layer is needed
        A, _ = self.forward_prop(X)

        # Calculate the cost using true labels and final predictions
        cost = self.cost(Y, A)

        # Convert probabilities to binary predictions (1 or 0)
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost
