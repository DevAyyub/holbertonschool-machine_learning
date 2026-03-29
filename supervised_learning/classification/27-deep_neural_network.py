#!/usr/bin/env python3
"""
DeepNeuralNetwork module for multiclass classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing multiclass classification
    """

    def __init__(self, nx, layers):
        """
        Class constructor
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            w_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)

            if i == 0:
                self.__weights[w_key] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[w_key] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
            
            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Updated for multiclass: Softmax on the output layer
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            A_prev = self.__cache['A' + str(i)]
            
            Z = np.matmul(W, A_prev) + b
            
            if i == self.__L - 1:
                # Softmax activation for the output layer
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability trick
                self.__cache['A' + str(i + 1)] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Sigmoid activation for hidden layers
                self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Z))
                
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost using categorical cross-entropy
        Y: one-hot (classes, m)
        A: softmax output (classes, m)
        """
        m = Y.shape[1]
        # categorical cross-entropy: -1/m * sum(Y * log(A))
        # Adding a small epsilon prevents log(0) errors
        loss = -np.sum(Y * np.log(A + 1e-8))
        cost = (1 / m) * loss
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        Returns: prediction (one-hot), cost
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        
        # Identify index of max probability for each example
        max_indices = np.argmax(A, axis=0)
        
        # Convert back to one-hot encoding
        prediction = np.eye(A.shape[0])[max_indices].T
        
        return prediction.astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        dz = cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]

            dw = (1 / m) * np.matmul(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                # Deriv of Sigmoid: A * (1 - A)
                da = np.matmul(W.T, dz)
                dz = da * (A_prev * (1 - A_prev))

            self.__weights['W' + str(i)] = W - (alpha * dw)
            self.__weights['b' + str(i)] = b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        costs = []
        iters = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                curr_cost = self.cost(Y, A)
                costs.append(curr_cost)
                iters.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, curr_cost))
            
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        import pickle
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        import pickle
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
