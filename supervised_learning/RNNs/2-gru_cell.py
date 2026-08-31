#!/usr/bin/env python3
"""GRU cell module."""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit."""

    def __init__(self, i, h, o):
        """Initialize the GRU cell.

        Args:
            i: Dimensionality of the input data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the output.
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: Previous hidden state of shape (m, h).
            x_t: Input data of shape (m, i).

        Returns:
            h_next: Next hidden state.
            y: Output probabilities.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = 1 / (
            1 + np.exp(
                -(np.matmul(concat, self.Wz) + self.bz)
            )
        )

        r = 1 / (
            1 + np.exp(
                -(np.matmul(concat, self.Wr) + self.br)
            )
        )

        concat_reset = np.concatenate(
            (r * h_prev, x_t),
            axis=1
        )

        h_inter = np.tanh(
            np.matmul(concat_reset, self.Wh) + self.bh
        )

        h_next = (1 - z) * h_prev + z * h_inter

        logits = np.matmul(h_next, self.Wy) + self.by

        exp = np.exp(
            logits - np.max(logits, axis=1, keepdims=True)
        )

        y = exp / np.sum(
            exp,
            axis=1,
            keepdims=True
        )

        return h_next, y
