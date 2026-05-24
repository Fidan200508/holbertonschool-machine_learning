#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    Args:
        X: numpy.ndarray of shape (n, d)
        k: number of clusters

    Returns:
        pi, m, S or None, None, None on failure
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0):
        return None, None, None

    n, d = X.shape

    pi = np.full(k, 1 / k)

    m, _ = kmeans(X, k)

    if m is None:
        return None, None, None

    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
