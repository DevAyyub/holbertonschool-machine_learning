#!/usr/bin/env python3
"""
Module to update weights with L2 Regularization using Gradient Descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization
    Args:
        Y: one-hot numpy.ndarray (classes, m) with correct labels
        weights: dictionary of weights and biases
        cache: dictionary of the outputs of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers of the network
    Returns: Nothing (Updates weights in place)
    """
    m = Y.shape[1]
    # Initialize dZ for the output layer (Softmax)
    # dZ = A[L] - Y
    dz = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        # A[i-1] is the input to the current layer
        A_prev = cache["A" + str(i - 1)]
        W_key = "W" + str(i)
        b_key = "b" + str(i)

        # Calculate Gradients
        # dW = (1/m) * (dz . A_prev.T) + (lambtha/m) * W
        dw = (np.matmul(dz, A_prev.T) / m) + (lambtha / m * weights[W_key])
        # db = (1/m) * sum(dz)
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            # Backpropagate dz to the previous layer
            # dz_prev = (W.T . dz) * derivative_of_tanh(A_prev)
            # derivative_of_tanh(A) = 1 - A^2
            W = weights[W_key]
            dz = np.matmul(W.T, dz) * (1 - (A_prev ** 2))

        # Update weights and biases in place
        weights[W_key] = weights[W_key] - alpha * dw
        weights[b_key] = weights[b_key] - alpha * db
