#!/usr/bin/env python3
"""Maximization step for GMM"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    Returns:
        pi, m, S or None, None, None on failure
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(g, np.ndarray) or len(g.shape) != 2):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    Nk = np.sum(g, axis=1)

    if np.any(Nk == 0):
        return None, None, None

    pi = Nk / n

    m = (g @ X) / Nk[:, np.newaxis]

    S = np.zeros((k, d, d))

    for i in range(k):
        X_m = X - m[i]
        S[i] = (g[i, :, np.newaxis] * X_m).T @ X_m / Nk[i]

    return pi, m, S
