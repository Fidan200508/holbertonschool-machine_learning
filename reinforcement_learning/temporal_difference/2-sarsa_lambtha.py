#!/usr/bin/env python3
"""Performs the SARSA(lambda) algorithm."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose an action using epsilon-greedy."""
    p = np.random.uniform()

    if p < epsilon:
        return np.random.randint(Q.shape[1])

    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """Perform the SARSA(lambda) algorithm."""
    initial_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_action = epsilon_greedy(
                Q,
                next_state,
                epsilon
            )

            delta = (
                reward
                + gamma * Q[next_state, next_action]
                - Q[state, action]
            )

            eligibility[state, action] += 1

            Q += alpha * delta * eligibility

            eligibility *= gamma * lambtha

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        epsilon = min_epsilon + (
            initial_epsilon - min_epsilon
        ) * np.exp(-epsilon_decay * episode)

    return Q
