#!/usr/bin/env python3
"""
t-SNE Module
"""
import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation:
    - X: numpy.ndarray of shape (n, d)
    - ndims: new dimensional representation of X
    - idims: intermediate dimensional representation of X after PCA
    - perplexity: perplexity value
    - iterations: number of iterations
    - lr: learning rate
    Returns: Y, numpy.ndarray of shape (n, ndim) optimized representation
    """
    X_pca = pca(X, idims)
    n = X_pca.shape[0]

    P = P_affinities(X_pca, perplexity=perplexity)
    P = P * 4.0

    Y = np.random.randn(n, ndims) * 1e-4
    Y_prev = Y.copy()

    for i in range(1, iterations + 1):
        dY, Q = grads(Y, P)

        alpha = 0.5 if i <= 20 else 0.8
        Y_new = Y - lr * dY + alpha * (Y - Y_prev)

        Y_prev = Y.copy()
        Y = Y_new - np.mean(Y_new, axis=0)

        if i == 100:
            P = P / 4.0

        if i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))

    return Y
