#!/usr/bin/env python3
"""
Module that implements Policy Gradient functions for reinforcement learning.
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix using softmax.

    Args:
        matrix (np.ndarray): The state matrix of shape (1, n) or (m, n).
        weight (np.ndarray): The weight matrix of shape (n, a).

    Returns:
        np.ndarray: The softmax policy probability distribution.
    """
    exp = np.exp(matrix @ weight)
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and weight.

    Args:
        state (np.ndarray): Matrix representing the current observation
                            of the environment.
        weight (np.ndarray): Matrix of random weight.

    Returns:
        tuple: The action (int) and the gradient (np.ndarray).
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])

    dsoftmax = -probs.copy()
    dsoftmax[0, action] += 1
    grad = state.T @ dsoftmax

    return action, grad
