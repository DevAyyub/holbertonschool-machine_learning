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
        """Calculates the cost of the model."""
        m = Y.shape[1]
        loss_1 = Y * np.log(A)
        loss_2 = (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(loss_1 + loss_2)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent."""
        m = Y.shape[1]
        dz = cache["A{}".format(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(i - 1)]
            W_curr = self.__weights["W{}".format(i)]
            b_curr = self.__weights["b{}".format(i)]

            dw = np.matmul(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            if i > 1:
                dz = np.matmul(W_curr.T, dz) * (A_prev * (1 - A_prev))

            self.__weights["W{}".format(i)] = W_curr - (alpha * dw)
            self.__weights["b{}".format(i)] = b_curr - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network.

        Parameters:
        X (numpy.ndarray): Input data (nx, m).
        Y (numpy.ndarray): Correct labels (1, m).
        iterations (int): Number of training passes.
        alpha (float): Learning rate.

        Returns:
        The evaluation of the training data after training.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Single allowed loop for training iterations
        for _ in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
