#!/usr/bin/env python3
"""
Module implementing the TD(lambda) algorithm for value estimation.
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm to estimate the value function V.

    Parameters:
    - env: the gymnasium environment instance
    - V: numpy.ndarray of shape (s,) containing the value estimate
    - policy: function that takes in a state and returns the next action
    - lambtha: eligibility trace factor
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount rate

    Returns:
    - V: the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()[0]
        Et = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            Et = gamma * lambtha * Et
            Et[state] += 1.0

            V = V + alpha * delta * Et

            if terminated or truncated:
                break
            state = next_state

    return V
