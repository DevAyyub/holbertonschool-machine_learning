#!/usr/bin/env python3
"""
Module to conduct forward propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout
    Args:
        X: numpy.ndarray (nx, m) containing the input data
        weights: dictionary of the weights and biases
        L: number of layers in the network
        keep_prob: probability that a node will be kept
    Returns:
        A dictionary containing the outputs of each layer and the
        dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        # Retrieve weights and biases for current layer
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        # Linear Step
        Z = np.matmul(W, A_prev) + b

        if i < L:
            # Activation for hidden layers (tanh)
            A = np.tanh(Z)
            # Create and apply dropout mask
            # Using binomial/rand to generate 1s with probability keep_prob
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            D = D.astype(int)
            # Scale the result (Inverted Dropout)
            A = (A * D) / keep_prob
            cache['D' + str(i)] = D
            cache['A' + str(i)] = A
        else:
            # Activation for output layer (softmax)
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache['A' + str(i)] = A

    return cache
