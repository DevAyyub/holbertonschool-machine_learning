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

        states = [step[0] for step in episode]
        for i, (s, _) in enumerate(episode):
            if s not in states[:i]:
                G = sum(step[1] * (gamma ** idx)
                        for idx, step in enumerate(episode[i:]))
                V[s] = V[s] + alpha * (G - V[s])

    return V
