#!/usr/bin/env python3
"""
Module defining a single neuron performing binary classification.
"""

import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """Initializes the neuron."""
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Retrieves the weights vector."""
        return self.__W

    @property
    def b(self):
        """Retrieves the bias."""
        return self.__b

    @property
    def A(self):
        """Retrieves the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation."""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model."""
        m = Y.shape[1]
        loss_1 = Y * np.log(A)
        loss_2 = (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(loss_1 + loss_2)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions."""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron."""
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron.

        Parameters:
        X (numpy.ndarray): Array with shape (nx, m) input data.
        Y (numpy.ndarray): Array with shape (1, m) correct labels.
        iterations (int): Number of iterations to train over.
        alpha (float): The learning rate.

        Returns:
        The evaluation of the training data after iterations.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # The single allowed loop to train the neuron
        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
