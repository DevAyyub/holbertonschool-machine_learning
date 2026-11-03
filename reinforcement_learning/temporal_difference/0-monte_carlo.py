#!/usr/bin/env python3
"""
Module containing the Monte Carlo policy evaluation algorithm.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Parameters:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append([state, reward])
            if terminated or truncated:
                break
            state = next_state

        episode = np.array(episode, dtype=int)
        G = 0
        for i, step in enumerate(episode[::-1]):
            s, r = step
            G = gamma * G + r
            if s not in episode[:i, 0]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
