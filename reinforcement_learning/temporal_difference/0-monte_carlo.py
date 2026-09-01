#!/usr/bin/env python3
"""Performs Monte Carlo value estimation."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Update state-value estimates using the Monte Carlo algorithm."""
    for _ in range(episodes):
        state = env.reset()[0]
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        G = 0

        for i in range(len(episode) - 1, -1, -1):
            state, reward = episode[i]
            G = reward + gamma * G

            previous_states = [
                step[0] for step in episode[:i]
            ]

            if state not in previous_states:
                V[state] += alpha * (G - V[state])

    return V
