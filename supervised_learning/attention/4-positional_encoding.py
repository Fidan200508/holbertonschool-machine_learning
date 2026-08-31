#!/usr/bin/env python3
"""Positional encoding for Transformer models."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculate sinusoidal positional encodings.

    Args:
        max_seq_len: Maximum sequence length.
        dm: Model depth.

    Returns:
        A numpy array of shape (max_seq_len, dm).
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dimensions // 2)) / np.float64(dm)
    )

    angles = positions * angle_rates

    positional_encoding = np.zeros(
        (max_seq_len, dm)
    )

    positional_encoding[:, 0::2] = np.sin(
        angles[:, 0::2]
    )

    positional_encoding[:, 1::2] = np.cos(
        angles[:, 1::2]
    )

    return positional_encoding
