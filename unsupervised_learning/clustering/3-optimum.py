#!/usr/bin/env python3
"""Finds the optimum number of clusters"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    Args:
        X: numpy.ndarray of shape (n, d)
        kmin: minimum number of clusters
        kmax: maximum number of clusters
        iterations: max iterations for K-means

    Returns:
        results, d_vars or None, None
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(kmin, int) or kmin <= 0 or
            kmax is None or
            not isinstance(kmax, int) or kmax <= 0 or
            kmax <= kmin or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None:
            return None, None

        results.append((C, clss))
        variances.append(variance(X, C))

    d_vars = [variances[0] - v for v in variances]

    return results, d_vars
