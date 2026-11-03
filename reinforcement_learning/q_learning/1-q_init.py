#!/usr/bin/env python3
"""Module that initializes the Q-table for reinforcement learning."""
import numpy as np


def q_init(env):
    """Initializes the Q-table with zeros.

    Args:
        env: The FrozenLakeEnv instance.

    Returns:
        numpy.ndarray: The Q-table initialized to zeros with shape
        (observation_space.n, action_space.n).
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
