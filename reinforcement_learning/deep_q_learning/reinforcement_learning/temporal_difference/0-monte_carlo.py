#!/usr/bin/env python3
"""Module that performs the Monte Carlo algorithm for value estimation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for policy evaluation.

    Args:
        env: Environment instance
        V (numpy.ndarray): Shape (s,) containing the value estimate
        policy: Function that takes in a state and returns the next action
        episodes (int): Total number of episodes to train over
        max_steps (int): Maximum number of steps per episode
        alpha (float): Learning rate
        gamma (float): Discount rate

    Returns:
        numpy.ndarray: V, the updated value estimate
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
        visited_states = set()

        for state, reward in reversed(episode):
            G = gamma * G + reward
            if state not in visited_states:
                V[state] = V[state] + alpha * (G - V[state])
                visited_states.add(state)

    return V
