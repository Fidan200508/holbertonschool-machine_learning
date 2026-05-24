#!/usr/bin/env python3
"""Finds best number of clusters using BIC"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds best number of clusters for a GMM using BIC"""

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(kmin, int) or kmin <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0 or kmax - kmin < 1:
        return None, None, None, None

    log_likelihoods = []
    bics = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)
        bic = p * np.log(n) - 2 * log_l

        log_likelihoods.append(log_l)
        bics.append(bic)
        results.append((pi, m, S))

    log_likelihoods = np.array(log_likelihoods)
    bics = np.array(bics)

    best_index = np.argmin(bics)

    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, log_likelihoods, bics
