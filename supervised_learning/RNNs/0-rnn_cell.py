#!/usr/bin/env python3
"""Simple RNN cell module."""

import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Initialize the RNN cell.

        Args:
            i: Dimensionality of the input data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the output.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: Previous hidden state of shape (m, h).
            x_t: Input data at the current time step of shape (m, i).

        Returns:
            h_next: Next hidden state.
            y: Output probabilities.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(
            np.matmul(concat, self.Wh) + self.bh
        )

        logits = np.matmul(h_next, self.Wy) + self.by

        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, y
