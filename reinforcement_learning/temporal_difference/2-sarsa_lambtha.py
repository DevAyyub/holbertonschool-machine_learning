#!/usr/bin/env python3
"""
Module implementing the SARSA(lambda) algorithm for Q-table estimation.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Selects the next action using an epsilon-greedy policy.

    Parameters:
    - Q: numpy.ndarray of shape (s, a) containing the Q table
    - state: the current state
    - epsilon: threshold for exploration vs exploitation

    Returns:
    - action: the selected action index
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, int(Q.shape[1]))
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(lambda) algorithm to update the Q table.

    Parameters:
    - env: the gymnasium environment instance
    - Q: numpy.ndarray of shape (s, a) containing the Q table
    - lambtha: eligibility trace factor
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount rate
    - epsilon: initial threshold for epsilon greedy
    - min_epsilon: minimum value that epsilon should decay to
    - epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
    - Q: the updated Q table
    """
    init_epsilon = epsilon

    for i in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)
        Et = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = reward + gamma * Q[next_state, next_action] - Q[
                state, action]
            Et = gamma * lambtha * Et
            Et[state, action] += 1.0

            Q = Q + alpha * delta * Et

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        epsilon = (min_epsilon + (init_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * i))

    return Q
