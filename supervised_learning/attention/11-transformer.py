#!/usr/bin/env python3
"""Transformer network module."""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Complete Transformer network."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1
    ):
        """Initialize the Transformer model.

        Args:
            N: Number of encoder and decoder blocks.
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in feed-forward layers.
            input_vocab: Size of the input vocabulary.
            target_vocab: Size of the target vocabulary.
            max_seq_input: Maximum input sequence length.
            max_seq_target: Maximum target sequence length.
            drop_rate: Dropout rate.
        """
        super().__init__()

        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )

        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
        self,
        inputs,
        target,
        training,
        encoder_mask,
        look_ahead_mask,
        decoder_mask
    ):
        """Perform the forward pass of the Transformer.

        Args:
            inputs: Input token indices.
            target: Target token indices.
            training: Boolean indicating training mode.
            encoder_mask: Padding mask for the encoder.
            look_ahead_mask: Look-ahead mask for the decoder.
            decoder_mask: Padding mask for decoder cross-attention.

        Returns:
            Tensor of shape (batch, target_seq_len, target_vocab).
        """
        encoder_output = self.encoder(
            inputs,
            training,
            encoder_mask
        )

        decoder_output = self.decoder(
            target,
            encoder_output,
            training,
            look_ahead_mask,
            decoder_mask
        )

        return self.linear(decoder_output)
