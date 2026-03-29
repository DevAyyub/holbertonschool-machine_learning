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
        """
        Initializes the neuron.

        Parameters:
        nx (int): The number of input features given to the neuron.
        """
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
        """
        Calculates the forward propagation.

        Parameters:
        X (numpy.ndarray): Array with shape (nx, m) containing input data.

        Returns:
        The private attribute __A (the activated output).
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        Y (numpy.ndarray): Array with shape (1, m) containing correct labels.
        A (numpy.ndarray): Array with shape (1, m) containing activated output.

        Returns:
        The cost.
        """
        m = Y.shape[1]
        loss_1 = Y * np.log(A)
        loss_2 = (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(loss_1 + loss_2)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Parameters:
        X (numpy.ndarray): Array with shape (nx, m) containing input data.
        Y (numpy.ndarray): Array with shape (1, m) containing correct labels.

        Returns:
        The neuron's prediction and the cost of the network, respectively.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Parameters:
        X (numpy.ndarray): Array with shape (nx, m) containing input data.
        Y (numpy.ndarray): Array with shape (1, m) containing correct labels.
        A (numpy.ndarray): Array with shape (1, m) containing activated output.
        alpha (float): The learning rate.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)
