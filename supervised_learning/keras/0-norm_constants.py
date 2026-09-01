#!/usr/bin/env python3
"""Calculates normalization constants for a data matrix."""

import numpy as np


def normalization_constants(X):
    """Return the mean and standard deviation of each feature."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
