#!/usr/bin/env python3
"""Policy gradient functions."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy using a weight matrix."""
    z = np.matmul(matrix, weight)
    exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """Compute the Monte Carlo policy gradient."""
    probabilities = policy(state, weight)

    action = np.random.choice(
        len(probabilities),
        p=probabilities
    )

    one_hot = np.zeros(len(probabilities))
    one_hot[action] = 1

    gradient = np.outer(
        state,
        one_hot - probabilities
    )

    return action, gradient
