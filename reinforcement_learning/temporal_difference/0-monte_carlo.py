#!/usr/bin/env python3
"""Performs the Monte Carlo algorithm."""

import numpy as np


def game(env, state, policy, max_steps):
    """Generate an episode using the given policy."""
    episode = []

    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        episode.append((state, action, reward))

        if terminated or truncated:
            break

        state = next_state

    return np.array(episode, dtype=int)


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform the Monte Carlo algorithm."""
    for i in range(episodes):
        state = env.reset()[0]
        G = 0

        episode = game(env, state, policy, max_steps)

        for step in episode[::-1]:
            state, _, reward = step
            G = gamma * G + reward

            if state not in episode[:i, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
