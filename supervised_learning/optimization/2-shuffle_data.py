#!/usr/bin/env python3
"""Shuffles two data matrices in the same way."""

import numpy as np


def shuffle_data(X, Y):
    """Shuffle X and Y using the same permutation."""
    permutation = np.random.permutation(X.shape[0])

    return X[permutation], Y[permutation]
