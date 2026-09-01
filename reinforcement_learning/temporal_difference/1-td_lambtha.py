#!/usr/bin/env python3
"""Performs the TD(lambda) algorithm."""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Update state-value estimates using TD(lambda)."""
    for _ in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]

            eligibility[state] += 1

            V += alpha * delta * eligibility

            eligibility *= gamma * lambtha

            state = next_state

            if terminated or truncated:
                break

    return V
