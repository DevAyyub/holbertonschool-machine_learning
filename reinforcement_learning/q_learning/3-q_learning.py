#!/usr/bin/env python3
"""Module that performs Q-learning on FrozenLake."""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Trains an agent using the Q-learning algorithm.

    Args:
        env: The FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.
        epsilon: Initial exploration rate.
        min_epsilon: Minimum value that epsilon can decay to.
        epsilon_decay: Decay rate for updating epsilon between episodes.

    Returns:
        tuple: (Q, total_rewards)
            Q is the updated Q-table.
            total_rewards is a list of rewards per episode.
    """
    initial_epsilon = epsilon
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            step_res = env.step(action)

            if len(step_res) == 5:
                next_state, reward, terminated, truncated, _ = step_res
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_res
                terminated = done

            if terminated and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            episode_reward += reward

            if done:
                break

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )
        total_rewards.append(episode_reward)

    return Q, total_rewards
