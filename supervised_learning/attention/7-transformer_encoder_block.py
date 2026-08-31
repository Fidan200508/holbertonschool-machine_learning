#!/usr/bin/env python3
"""Transformer encoder block module."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Create one Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the encoder block.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in the feed-forward layer.
            drop_rate: Dropout rate.
        """
        super().__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )

        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Perform the forward pass of the encoder block.

        Args:
            x: Tensor of shape (batch, input_seq_len, dm).
            training: Boolean indicating training mode.
            mask: Optional attention mask.

        Returns:
            Tensor of shape (batch, input_seq_len, dm).
        """
        attention, _ = self.mha(
            x,
            x,
            x,
            mask
        )

        attention = self.dropout1(
            attention,
            training=training
        )

        out1 = self.layernorm1(
            x + attention
        )

        hidden = self.dense_hidden(out1)
        output = self.dense_output(hidden)

        output = self.dropout2(
            output,
            training=training
        )

        return self.layernorm2(
            out1 + output
        )
