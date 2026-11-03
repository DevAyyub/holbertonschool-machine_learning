#!/usr/bin/env python3
"""Module that has a trained agent play an episode in FrozenLake."""
import numpy as np


def play(env, Q, max_steps=100):
    """Has the trained agent play an episode using greedy exploitation.

    Args:
        env: The FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        max_steps: Maximum number of steps in the episode.

    Returns:
        tuple: (total_rewards, rendered_outputs)
            total_rewards is the total reward for the episode.
            rendered_outputs is a list of rendered strings representing
            the board state at each step.
    """
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    rendered_outputs = [env.render()]
    total_rewards = 0

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        step_res = env.step(action)

        if len(step_res) == 5:
            next_state, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_res

        rendered_outputs.append(env.render())
        total_rewards += reward
        state = next_state

        if done:
            break

    return total_rewards, rendered_outputs
