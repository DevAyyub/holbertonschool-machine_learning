#!/usr/bin/env python3
"""
Module containing the Monte Carlo policy evaluation algorithm.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the first-visit Monte Carlo value estimation algorithm.

    Parameters:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function taking a state and returning an action
        episodes: total number of episodes to train over
        max_steps: maximum steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        G = 0
        visited = set()
        for s, r in reversed(episode):
            G = gamma * G + r
            if s not in visited:
                visited.add(s)
                V[s] = V[s] + alpha * (G - V[s])

    return V
