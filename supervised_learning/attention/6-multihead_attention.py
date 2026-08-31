#!/usr/bin/env python3
"""Multi-head attention module."""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi-head attention."""

    def __init__(self, dm, h):
        """Initialize multi-head attention.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
        """
        super().__init__()

        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch):
        """Split the last dimension into multiple heads."""
        x = tf.reshape(
            x,
            (batch, -1, self.h, self.depth)
        )

        return tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

    def call(self, Q, K, V, mask):
        """Perform multi-head attention.

        Args:
            Q: Query input tensor.
            K: Key input tensor.
            V: Value input tensor.
            mask: Optional attention mask.

        Returns:
            output: Multi-head attention output.
            weights: Attention weights for all heads.
        """
        batch = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch)
        K = self.split_heads(K, batch)
        V = self.split_heads(V, batch)

        attention, weights = sdp_attention(
            Q,
            K,
            V,
            mask
        )

        attention = tf.transpose(
            attention,
            perm=[0, 2, 1, 3]
        )

        attention = tf.reshape(
            attention,
            (batch, -1, self.dm)
        )

        output = self.linear(attention)

        return output, weights
