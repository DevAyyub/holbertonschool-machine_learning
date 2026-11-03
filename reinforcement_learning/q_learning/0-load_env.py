#!/usr/bin/env python3
"""Module that loads the FrozenLake environment from Gymnasium."""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Args:
        desc: List of lists containing a custom map description, or None.
        map_name: Pre-made map name string ('4x4' or '8x8'), or None.
        is_slippery: Boolean indicating if ice is slippery.

    Returns:
        The gymnasium FrozenLake environment.
    """
    if desc is None and map_name is None:
        map_name = "8x8"

    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
