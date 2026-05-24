#!/usr/bin/env python3
"""Finds the best number of clusters using BIC"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC

    Returns:
        best_k, best_result, l, b or None, None, None, None
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(kmin, int) or kmin <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0 or kmax <= kmin:
        return None, None, None, None

    l = []
    b = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)

        bic = p * np.log(n) - 2 * log_l

        l.append(log_l)
        b.append(bic)
        results.append((pi, m, S))

    l = np.array(l)
    b = np.array(b)

    index = np.argmin(b)

    best_k = kmin + index
    best_result = results[index]

    return best_k, best_result, l, b
