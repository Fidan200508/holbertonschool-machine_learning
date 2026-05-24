#!/usr/bin/env python3
"""Expectation step for GMM"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    Returns:
        g, l or None, None on failure
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(pi, np.ndarray) or len(pi.shape) != 1 or
            not isinstance(m, np.ndarray) or len(m.shape) != 2 or
            not isinstance(S, np.ndarray) or len(S.shape) != 3):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None

    weighted = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])

        if P is None:
            return None, None

        weighted[i] = pi[i] * P

    total = np.sum(weighted, axis=0)

    if np.any(total == 0):
        return None, None

    g = weighted / total

    l = np.sum(np.log(total))

    return g, l
