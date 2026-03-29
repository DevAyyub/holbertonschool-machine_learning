#!/usr/bin/env python3
"""
DeepNeuralNetwork module with configurable hidden layer activations
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing multiclass classification
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Class constructor
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)

            if i == 0:
                self.__weights[w_key] = (np.random.randn(layers[i], nx) *
                                         np.sqrt(2 / nx))
            else:
                self.__weights[w_key] = (np.random.randn(layers[i],
                                         layers[i-1]) *
                                         np.sqrt(2 / layers[i-1]))

            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    @property
    def activation(self):
        """Getter for activation"""
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            A_prev = self.__cache['A' + str(i)]

            Z = np.dot(W, A_prev) + b

            if i == self.__L - 1:
                # Softmax for output layer
                t = np.exp(Z)
                self.__cache['A' + str(i + 1)] = (t / np.sum(t, axis=0,
                                                  keepdims=True))
            else:
                # Configurable hidden layer activation
                if self.__activation == 'sig':
                    self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Z))
                else:
                    self.__cache['A' + str(i + 1)] = np.tanh(Z)

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates categorical cross-entropy cost"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates prediction and cost"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        max_indices = np.argmax(A, axis=0)
        classes = A.shape[0]
        prediction = np.eye(classes)[max_indices].T
        return prediction.astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent"""
        m = Y.shape[1]
        dz = cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]

            dw = (1 / m) * np.dot(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                # Backprop through the chosen hidden activation
                if self.__activation == 'sig':
                    g_prime = A_prev * (1 - A_prev)
                else:
                    g_prime = 1 - (A_prev ** 2)
                dz = np.dot(W.T, dz) * g_prime

            self.__weights['W' + str(i)] = W - (alpha * dw)
            self.__weights['b' + str(i)] = b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        costs = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                curr_cost = self.cost(Y, A)
                costs.append(curr_cost)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, curr_cost))

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves instance to file"""
        import pickle
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads instance from file"""
        import pickle
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
