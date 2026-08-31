#!/usr/bin/env python3
"""Calculate the marginal probability of obtaining data."""

import numpy as np

intersection = __import__('1-intersection').intersection


def marginal(x, n, P, Pr):
    """Calculate the marginal probability of obtaining x and n."""
    intersections = intersection(x, n, P, Pr)

    return np.sum(intersections)
