#!/usr/bin/env python3
"""Forward propagation for a deep RNN."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells: List of RNNCell instances.
        X: Input data of shape (t, m, i).
        h_0: Initial hidden states of shape (l, m, h).

    Returns:
        H: All hidden states.
        Y: Outputs from the final RNN layer.
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        layer_input = X[step]

        for layer in range(l):
            h_next, y = rnn_cells[layer].forward(
                H[step, layer],
                layer_input
            )

            H[step + 1, layer] = h_next
            layer_input = h_next

        Y[step] = y

    return H, Y
