#!/usr/bin/env python3
"""
Module to train a DQN agent to play Atari Breakout using keras-rl2.
"""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import keras
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers.legacy import Adam
import tensorflow as tf

tf.keras.__version__ = keras.__version__

from rl.agents.dqn import DQNAgent  # noqa: E402
from rl.memory import SequentialMemory  # noqa: E402
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy  # noqa: E402


class GymnasiumToKerasRLWrapper(gym.Wrapper):
    """
    Wrapper to adapt Gymnasium API changes to legacy Gym/Keras-RL2 API.
    """

    def __init__(self, env):
        """
        Initialize the wrapper.

        Parameters:
            env: Gymnasium environment to wrap.
        """
        super().__init__(env)
        self.env = env

    def reset(self, **kwargs):
        """
        Reset environment returning observation only.

        Parameters:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: Initial observation.
        """
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
            action: Action to take.

        Returns:
            tuple: (observation, reward, done, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info


def build_model(input_shape, nb_actions):
    """
    Build the convolutional neural network for DQN.

    Parameters:
        input_shape: Shape of the input window (window_length, height, width).
        nb_actions: Number of available actions.

    Returns:
        keras.Model: Sequential Keras model.
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def train():
    """
    Train the DQN agent and save the policy network.
    """
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True
    )
    env = GymnasiumToKerasRLWrapper(env)

    nb_actions = env.action_space.n
    window_length = 4
    input_shape = (window_length, 84, 84)

    model = build_model(input_shape, nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=window_length)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    dqn.fit(env, nb_steps=1750000, log_interval=10000, visualize=False)

    model.save('policy.h5')


if __name__ == '__main__':
    train()
