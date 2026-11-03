#!/usr/bin/env python3
"""
Module that implements the training process for Policy Gradient.
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full Monte-Carlo policy gradient training.

    Args:
        env: Initial gymnasium environment.
        nb_episodes (int): Number of episodes used for training.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        show_result (bool): Whether to render the environment every
                            1000 episodes.

    Returns:
        list: All values of the score (sum of all rewards during one
              episode loop).
    """
    weight = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        done = False

        if show_result and episode % 1000 == 0:
            env.render()

        while not done:
            action, grad = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            grads.append(grad)
            rewards.append(reward)

        score = sum(rewards)
        scores.append(score)
        print(f"Episode: {episode} Score: {score}", end="\r", flush=False)

        for i in range(len(grads)):
            gt = sum(
                rewards[t] * (gamma ** (t - i))
                for t in range(i, len(rewards))
            )
            weight += alpha * grads[i] * gt

    return scores
