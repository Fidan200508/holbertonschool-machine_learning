#!/usr/bin/env python3
"""K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    Args:
        X: numpy.ndarray of shape (n, d)
        k: number of clusters
        iterations: maximum number of iterations

    Returns:
        C, clss or None, None on failure
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    C = np.random.uniform(min_vals, max_vals, size=(k, d))

    for _ in range(iterations):
        C_prev = C.copy()

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for i in range(k):
            points = X[clss == i]

            if points.shape[0] == 0:
                C[i] = np.random.uniform(min_vals, max_vals)
            else:
                C[i] = np.mean(points, axis=0)

        if np.array_equal(C, C_prev):
            return C, clss

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
