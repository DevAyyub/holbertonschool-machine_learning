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
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize weights with random normal distribution, shape (1, nx)
        self.W = np.random.randn(1, nx)
        # Initialize bias to 0
        self.b = 0
        # Initialize activated output to 0
        self.A = 0
