#!/usr/bin/env python3
"""Transformer decoder block module."""

import tensorflow as tf

MultiHeadAttention = __import__(
    '6-multihead_attention'
).MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Create one Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the decoder block.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in the feed-forward layer.
            drop_rate: Dropout rate.
        """
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

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

        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(
        self,
        x,
        encoder_output,
        training,
        look_ahead_mask,
        padding_mask
    ):
        """Perform the forward pass of the decoder block.

        Args:
            x: Tensor of shape (batch, target_seq_len, dm).
            encoder_output: Encoder output tensor.
            training: Boolean indicating training mode.
            look_ahead_mask: Mask for the first attention layer.
            padding_mask: Mask for the second attention layer.

        Returns:
            Tensor of shape (batch, target_seq_len, dm).
        """
        attention1, _ = self.mha1(
            x,
            x,
            x,
            look_ahead_mask
        )

        attention1 = self.dropout1(
            attention1,
            training=training
        )

        out1 = self.layernorm1(
            x + attention1
        )

        attention2, _ = self.mha2(
            out1,
            encoder_output,
            encoder_output,
            padding_mask
        )

        attention2 = self.dropout2(
            attention2,
            training=training
        )

        out2 = self.layernorm2(
            out1 + attention2
        )

        hidden_output = self.dense_hidden(out2)
        output = self.dense_output(hidden_output)

        output = self.dropout3(
            output,
            training=training
        )

        return self.layernorm3(
            out2 + output
        )
