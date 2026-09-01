#!/usr/bin/env python3
"""Loads the FrozenLake environment."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load and return a FrozenLake environment."""
    if desc is None and map_name is None:
        desc = gym.envs.toy_text.frozen_lake.generate_random_map(size=8)

    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )

    return env
