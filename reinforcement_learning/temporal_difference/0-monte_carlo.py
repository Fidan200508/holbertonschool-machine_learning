#!/usr/bin/env python3
"""Performs the Monte Carlo algorithm."""

import numpy as np


def generate_episode(env, policy, max_steps):
    """Generate one episode following a policy."""
    episode = [[], []]
    state = env.reset()[0]
    desc = env.unwrapped.desc.reshape(env.observation_space.n)

    for _ in range(max_steps):
        action = policy(state)
        next_state, _, terminated, truncated, _ = env.step(action)

        episode[0].append(state)

        if desc[next_state] == b'H':
            episode[1].append(-1)
            break

        if desc[next_state] == b'G':
            episode[1].append(1)
            break

        episode[1].append(0)
        state = next_state

        if terminated or truncated:
            break

    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform the Monte Carlo algorithm."""
    discounts = np.array([gamma ** i for i in range(max_steps)])

    for _ in range(episodes):
        episode = generate_episode(env, policy, max_steps)

        for i in range(len(episode[0])):
            rewards = np.array(episode[1][i:])
            discount = discounts[:len(rewards)]
            G = np.sum(rewards * discount)

            state = episode[0][i]
            V[state] += alpha * (G - V[state])

    return V
