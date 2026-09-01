#!/usr/bin/env python3
"""Calculates normalization constants."""

import numpy as np


def normalization_constants(X):
    """Calculate mean and standard deviation for each feature."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
