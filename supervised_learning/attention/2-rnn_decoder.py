#!/usr/bin/env python3
"""RNN decoder module for machine translation."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN decoder using attention and a GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNN decoder.

        Args:
            vocab: Size of the target vocabulary.
            embedding: Dimensionality of embedding vectors.
            units: Number of hidden units in the GRU.
            batch: Batch size.
        """
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(
            vocab,
            embedding
        )

        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        self.F = tf.keras.layers.Dense(vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Perform the forward pass of the decoder.

        Args:
            x: Previous target word of shape (batch, 1).
            s_prev: Previous decoder hidden state.
            hidden_states: Encoder outputs.

        Returns:
            y: Output logits of shape (batch, vocab).
            s: New decoder hidden state.
        """
        context, weights = self.attention(
            s_prev,
            hidden_states
        )

        x = self.embedding(x)

        context = tf.expand_dims(
            context,
            axis=1
        )

        x = tf.concat(
            [context, x],
            axis=-1
        )

        output, s = self.gru(x)

        output = tf.reshape(
            output,
            (-1, output.shape[2])
        )

        y = self.F(output)

        return y, s
