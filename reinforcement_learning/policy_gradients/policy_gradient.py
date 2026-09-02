#!/usr/bin/env python3
"""Policy gradient functions."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy using a weight matrix."""
    z = np.matmul(matrix, weight)
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)
