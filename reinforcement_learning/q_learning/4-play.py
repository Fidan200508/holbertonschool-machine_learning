#!/usr/bin/env python3
"""Allows a trained Q-learning agent to play an episode."""

import numpy as np


def play(env, Q, max_steps=100):
    """Play an episode by always exploiting the Q-table."""
    state = env.reset()[0]
    total_rewards = 0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = np.argmax(Q[state])

        state, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
