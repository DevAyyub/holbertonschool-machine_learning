#!/usr/bin/env python3
"""
Module implementing the Monte Carlo algorithm for value estimation.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function V.

    Parameters:
    - env: the gymnasium environment instance
    - V: numpy.ndarray of shape (s,) containing the value estimate
    - policy: function that takes in a state and returns the next action
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount rate

    Returns:
    - V: the updated value estimate
    """
    for i in range(episodes):
        state = env.reset()[0]
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        episode = np.array(episode, dtype=int)
        G = 0

        for j in range(len(episode) - 1, -1, -1):
            state, reward = episode[j]
            G = gamma * G + reward
            if state not in episode[:i, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
