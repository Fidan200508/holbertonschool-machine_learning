#!/usr/bin/env python3
"""Initialize K-means centroids"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Parameters:
    X : numpy.ndarray of shape (n, d)
        Dataset
    k : int
        Number of clusters

    Returns:
    numpy.ndarray of shape (k, d) containing initialized centroids
    or None on failure
    """

    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(k, int) or
            k <= 0):
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    return np.random.uniform(
        low=min_vals,
        high=max_vals,
        size=(k, X.shape[1])
    )
