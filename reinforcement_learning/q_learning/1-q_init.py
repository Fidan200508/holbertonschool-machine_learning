#!/usr/bin/env python3
"""Initializes the Q-table."""

import numpy as np


def q_init(env):
    """Initialize and return a Q-table of zeros."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    return np.zeros((n_states, n_actions))
