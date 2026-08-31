#!/usr/bin/env python3
"""Calculate the likelihood of binomial data."""

import numpy as np


def likelihood(x, n, P):
    """Calculate likelihoods for different probabilities."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError(
            "All values in P must be in the range [0, 1]"
        )

    def factorial(num):
        """Calculate factorial."""
        result = 1

        for i in range(1, num + 1):
            result *= i

        return result

    combination = (
        factorial(n)
        / (
            factorial(x)
            * factorial(n - x)
        )
    )

    return (
        combination
        * (P ** x)
        * ((1 - P) ** (n - x))
    )
