#!/usr/bin/env python3
"""
Module defining a single neuron for binary classification.
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
        nx (int): The number of input features to the neuron.
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
        """Getter for the weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for the bias."""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): Array with shape (nx, m) containing the input data.
                           nx is the number of input features to the neuron.
                           m is the number of examples.

        Returns:
        The private attribute __A (the activated output).
        """
        Z = np.matmul(self.__W, X) + self.__b

        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A
