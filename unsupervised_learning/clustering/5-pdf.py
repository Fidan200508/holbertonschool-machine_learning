#!/usr/bin/env python3
"""Calculates PDF of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    Args:
        X: numpy.ndarray of shape (n, d)
        m: numpy.ndarray of shape (d,)
        S: numpy.ndarray of shape (d, d)

    Returns:
        P or None on failure
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(m, np.ndarray) or len(m.shape) != 1 or
            not isinstance(S, np.ndarray) or len(S.shape) != 2):
        return None

    n, d = X.shape

    if m.shape[0] != d or S.shape != (d, d):
        return None

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)

        X_m = X - m

        exponent = -0.5 * np.sum((X_m @ inv) * X_m, axis=1)

        denominator = np.sqrt(((2 * np.pi) ** d) * det)

        P = (1 / denominator) * np.exp(exponent)

        P = np.maximum(P, 1e-300)

        return P

    except Exception:
        return None
