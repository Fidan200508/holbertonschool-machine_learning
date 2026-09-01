#!/usr/bin/env python3
"""Selects an action using the epsilon-greedy policy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose the next action using epsilon-greedy."""
    p = np.random.uniform()

    if p < epsilon:
        return np.random.randint(Q.shape[1])

    return np.argmax(Q[state])
