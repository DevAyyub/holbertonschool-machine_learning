#!/usr/bin/env python3
"""Module that determines the next action using epsilon-greedy."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action.

    Args:
        Q: numpy.ndarray containing the q-table.
        state: Current state index.
        epsilon: Epsilon value to determine exploration vs exploitation.

    Returns:
        int: The next action index.
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action
