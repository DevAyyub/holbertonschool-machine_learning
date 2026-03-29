#!/usr/bin/env python3
"""
Module defining a deep neural network with persistence methods.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


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

        # Loop 1: Layer initialization
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
        """Retrieves layer count."""
        return self.__L

    @property
    def cache(self):
        """Retrieves cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Retrieves weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation."""
        self.__cache["A0"] = X
        # Loop 2: Propagation through layers
        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A{}".format(i)] = A
        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates cost of the model."""
        m = Y.shape[1]
        loss_1 = Y * np.log(A)
        loss_2 = (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(loss_1 + loss_2)
        return cost

    def evaluate(self, X, Y):
        """Evaluates network predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates gradient descent."""
        m = Y.shape[1]
        dz = cache["A{}".format(self.__L)] - Y
        # Loop 3: Backpropagation
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

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep network."""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        steps, costs = [], []
        # Loop 4: Training iterations
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                steps.append(i)
                costs.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object."""
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
