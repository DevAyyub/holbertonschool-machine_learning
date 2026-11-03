#!/usr/bin/env python3
"""
Module to load and display Atari Breakout played by a trained DQN agent.
"""
import sys
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import keras
import keras.api._v2.keras as keras_v2
from keras.models import load_model
from keras.optimizers.legacy import Adam
import tensorflow as tf

keras_v2.__version__ = keras.__version__
sys.modules['tensorflow.keras'] = keras_v2

from rl.agents.dqn import DQNAgent  # noqa: E402
from rl.memory import SequentialMemory  # noqa: E402
from rl.policy import GreedyQPolicy  # noqa: E402


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

    def render(self, *args, **kwargs):
        """
        Render the environment compatible with keras-rl2.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Rendered frame or None.
        """
        return self.env.render()


def play():
    """
    Load saved policy and run visual evaluation.
    """
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
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

    model = load_model('policy.h5')

    memory = SequentialMemory(limit=100000, window_length=window_length)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()


if __name__ == '__main__':
    play()
