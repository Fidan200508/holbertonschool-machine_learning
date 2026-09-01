#!/usr/bin/env python3
"""Normalizes a data matrix."""

import numpy as np


def normalize(X, m, s):
    """Normalize X using the supplied mean and standard deviation."""
    return (X - m) / s
