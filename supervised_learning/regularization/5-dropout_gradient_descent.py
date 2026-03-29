#!/usr/bin/env python3
"""
Module to update weights with Dropout Regularization
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    Args:
        Y: one-hot numpy.ndarray (classes, m) containing correct labels
        weights: dictionary of the weights and biases
        cache: dictionary of the outputs and dropout masks of each layer
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network
    Returns: Nothing (Updates weights in place)
    """
    m = Y.shape[1]
    # dZ for output layer (Softmax)
    dz = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        W_key = "W" + str(i)
        b_key = "b" + str(i)
        W = weights[W_key]

        # Calculate Gradients
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            # Backpropagate to previous layer
            da_prev = np.matmul(W.T, dz)
            # Apply mask and scale (Inverted Dropout)
            mask = cache["D" + str(i - 1)]
            da_prev = (da_prev * mask) / keep_prob
            # dZ for tanh layer
            dz = da_prev * (1 - (A_prev ** 2))

        # Update weights and biases in place
        weights[W_key] -= alpha * dw
        weights[b_key] -= alpha * db
