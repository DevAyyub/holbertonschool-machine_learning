#!/usr/bin/env python3
"""
Module defining a neural network with one hidden layer.
"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing
    binary classification.
    """

    def __init__(self, nx, nodes):
        """Initializes the neural network."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Retrieves the weights vector of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Retrieves the bias of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Retrieves the activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Retrieves the weights vector of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Retrieves the bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Retrieves the activated output of the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        m = Y.shape[1]
        loss_1 = Y * np.log(A)
        loss_2 = (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(loss_1 + loss_2)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost
